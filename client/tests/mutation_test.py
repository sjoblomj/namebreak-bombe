#!/usr/bin/env python3
"""Mutation testing of the backends' search code.

Builds copies of the client in which a backend's search is broken on purpose -
each mutation a small bug that would make a search silently miss (or invent)
candidates - and checks that every one of them is caught by each of the
three checks that guard the search, on its own:

  self-test          the known-answer test createBackend runs on every
                     backend (src/backends/self_test.h)
  integration        tests/search_integration_test.cpp
  stress             tests/search_stress_test.cpp

Two more run on every copy too. tests/search_overflow_test.cpp
("overflow") covers what happens when a launch has more hits than it can
record, which the other three rarely or never reach. And the stress test
built for tiny batches, a few searched per launch ("stress-small" - the
small_launch variant in CMakeLists.txt): a CUDA launch covers 16 leading
values, so in the default build nearly every test search is a single launch,
and only this one searches with many - where what one launch leaves behind
can break the next. A mutation there is expected to be caught by those
alone; and the engine's mutations can't be caught by the self-test, which
tests a backend without the engine. Each mutation names the checks that must
catch it (the first three, by default).

In the copies, createBackend's own self-test is switched off, so that the
integration and stress tests are judged by themselves; the self-test is run
separately instead. Two kinds of run check the experiment itself: the
unmodified code must pass all five checks (otherwise nothing here means
anything, and the script stops), and so must the mutations marked harmless -
changes that must *not* change what gets searched.

One backend per run (--backend): cuda (the default), hip (the same code,
compiled with HIP), opencl, metal (on a Mac) or cpu, each with its own list
of mutations (--list).
Needs the GPU the backend runs on, its toolchain or driver, and CMake. It
copies CMakeLists.txt, src/ and tests/ as they are on disk into a work
directory, so it tests uncommitted changes too and never touches build/.
The CUDA list takes about fifteen minutes on a laptop i9-12900H and RTX 3080
Ti.

Run from anywhere:  python3 client/tests/mutation_test.py [--backend opencl] [options]
or, from a configured build:  cmake --build --preset default --target run_mutation_test
                              (or run_mutation_test_opencl)

Exit status: 0 if every mutation was caught by the checks it names and every
control passed all five, 1 if not, 2 if the experiment couldn't be run.

A mutation is an exact string replacement in one file. When the code it
targets changes, the string no longer matches and the script says so -
update the mutation to the new code rather than dropping it.
"""

import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

CLIENT = Path(__file__).resolve().parent.parent
KERNEL = "src/backends/cuda/cuda_backend.cu"
FILTER = "src/backends/common/lowbits_filter.cpp"
ENGINE = "src/engine/search.cpp"
CL_KERNEL = "src/backends/opencl/search.cl"
CL_HOST = "src/backends/opencl/opencl_backend.cpp"
MTL_KERNEL = "src/backends/metal/search.metal"
MTL_HOST = "src/backends/metal/metal_backend.mm"
CPU = "src/backends/cpu/cpu_backend.cpp"
PRUNING = "src/backends/common/row_pruning.cpp"


@dataclass
class Mutation:
    name: str
    what: str
    # (file, old, new): `old` must occur exactly once in `file`.
    edits: list
    # "caught": every check in caught_by must fail. "harmless": every check
    # must pass.
    expect: str = "caught"
    caught_by: tuple = ("self-test", "integration", "stress")


# The table builder is shared by every backend that has the filter.
WRONG_TARGET = Mutation("wrongtarget", "the table built for the wrong target",
                        [(FILTER, "const Lane targetA = (Lane) constants.targetHashA;",
                          "const Lane targetA = (Lane) (constants.targetHashA ^ 1);")])

# The lists of row groups a search that prunes the whole candidate searches
# (backends/common/row_pruning.cpp), shared by the GPU backends.
ROW_PRUNING_MUTATIONS = [
    Mutation("groupsunpruned", "no row group left out of a list",
             [(PRUNING, "c.ok = pruneStep_CPU(rules_, c.state, alphabet_[d]) &&", "c.ok = true || pruneStep_CPU(rules_, c.state, alphabet_[d]) &&")]),
    Mutation("lastcapacity", "a row not left out when its last character can't make up min_backslash_count",
             [(PRUNING, "canReachMinBackslashes_CPU(rules_, s, capacityFrom[trailingLen - 1])", "true")]),
    Mutation("rowinserted", "the text inserted after a row's own character not checked",
             [(PRUNING, "stepOver(rules_, s, afterRow)", "true")]),
    Mutation("groupinserted", "the text inserted between a row group's characters not checked",
             [(PRUNING, "stepOver(rules_, c.state, insertedBefore(trailingLen, depth + 1)) &&", "")]),
    Mutation("closerclass", "')' taken for '(' when the rows a group's entry allows are worked out",
             [(PRUNING, "        case ')': return 3;", "        case ')': return 2;")]),
    Mutation("sliceend", "a batch's slice of its list one row group short at its end",
             [(PRUNING, "std::upper_bound(lo, end, lastGroup,", "std::upper_bound(lo, end, lastGroup - 1,")]),
    Mutation("slicestart", "a batch's slice of its list one row group short at its start",
             [(PRUNING, "std::lower_bound(begin, end, firstGroup,", "std::lower_bound(begin, end, firstGroup + 1,")]),
    # Only a search that leaves more brackets open before its trailing part
    # than that has characters to close them can tell.
    Mutation("bracketclamp", "an entry state's open brackets counted one fewer than they can matter",
             [(PRUNING, "std::min(entry.open.round, reach)", "std::min(entry.open.round, reach - 1)")],
             caught_by=("integration",)),
    # Only a backend that keeps its copy of the lists between searches can
    # tell; the self-test's cases have two alphabets, the others many.
    Mutation("nogeneration", "the lists cleared for another alphabet without a backend being told",
             [(PRUNING, "    lists_.clear();\n    ++generation_;", "    lists_.clear();")]),
]

# The engine's side of pruning the whole candidate: the self-test tests a
# backend without the engine.
ENGINE_PRUNING_MUTATIONS = [
    Mutation("wholeoff", "the backend never told to prune the whole candidate",
             [(ENGINE, "if (req.pruneWholeCandidate)\n        constants.trailingRules = rules;",
               "if (false)\n        constants.trailingRules = rules;")],
             caught_by=("integration", "stress")),
    Mutation("noentry", "every batch's rules started as if there were no leading characters",
             [(ENGINE, "params.pruneEntry = pruneEntry;", "params.pruneEntry = candidateStart;")],
             caught_by=("integration", "stress")),
]

CUDA_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(KERNEL, "const uint64_t alphabetMask = (uint64_t(1) << alphabetSize) - 1;", "const uint64_t alphabetMask = 0xFFFFFFFFull;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(KERNEL, "__ldg(&bufs.filterTable[lowBitsFilterIndex(seed1, seed2, kCudaLowBitsFilterBits)]);",
               "__ldg(&bufs.filterTable[lowBitsFilterIndex(seed2, seed1, kCudaLowBitsFilterBits)]);")]),
    Mutation("lastrow", "a launch's last row one candidate short",
             [(KERNEL, "const uint64_t lastRowMask = (uint64_t(1) << lastRowEndK) - 1;", "const uint64_t lastRowMask = (uint64_t(1) << (lastRowEndK - 1)) - 1;")]),
    Mutation("firstrow", "a launch's first row one candidate short",
             [(KERNEL, "alphabetMask & (~uint64_t(0) << firstRowStartK);", "alphabetMask & (~uint64_t(0) << (firstRowStartK + 1));")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(KERNEL, "mask &= (d == firstRowD) ? firstRowMask : alphabetMask;", "mask &= alphabetMask;"), (KERNEL, "if (d == lastRowD)", "if (false)")]),
    Mutation("skiplast", "only the first flagged candidate of a chunk's last flagged row hashed",
             [(KERNEL, "while ((flaggedRows | mask) != 0) {", "while (flaggedRows != 0) {")]),
    Mutation("flaggedrow", "a chunk's flagged rows' candidates hashed from the next row's state",
             [(KERNEL, "d = dBegin + __ffsll((long long) flaggedRows) - 1;", "d = dBegin + __ffsll((long long) flaggedRows);")]),
    Mutation("flaggedbit", "a row's flag recorded as the next row's",
             [(KERNEL, "flaggedRows |= (ChunkRowBits) (mask != 0) << (d - dBegin);", "flaggedRows |= (ChunkRowBits) (mask != 0) << (d - dBegin) << 1;")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(KERNEL, "#pragma unroll\n            for (int i = 0; i < SuffixLen; ++i)",
               "#pragma unroll\n            for (int i = 0; i + 1 < SuffixLen; ++i)")]),
    WRONG_TARGET,
    # Row groups and their chunks.
    Mutation("chunkend", "every chunk's last row skipped",
             [(KERNEL, "int dEnd = (int) divideByChunksPerGroup<FixedSize>((chunk + 1) * alphabetSize, shape);",
               "int dEnd = (int) divideByChunksPerGroup<FixedSize>((chunk + 1) * alphabetSize, shape) - 1;")]),
    Mutation("chunkstart", "every chunk's first row skipped",
             [(KERNEL, "int dBegin = (int) divideByChunksPerGroup<FixedSize>(chunk * alphabetSize, shape);",
               "int dBegin = (int) divideByChunksPerGroup<FixedSize>(chunk * alphabetSize, shape) + 1;")]),
    Mutation("firstclip", "the range cut one row short at its start",
             [(KERNEL, "        dBegin = firstRowD;", "        dBegin = firstRowD + 1;")]),
    Mutation("lastclip", "the range cut one row short at its end",
             [(KERNEL, "dEnd = lastRowD + 1;", "dEnd = lastRowD;")]),
    Mutation("firstrowd", "the first row's cut applied to the wrong row",
             [(KERNEL, "batch.firstRowD = (int) (firstRow % alphabetSize);", "batch.firstRowD = (int) (firstRow % alphabetSize) + 1;")]),
    Mutation("lastrowd", "the last row's cut applied to the wrong row",
             [(KERNEL, "batch.lastRowD = (int) (lastRow % alphabetSize);", "batch.lastRowD = (int) (lastRow % alphabetSize) - 1;")]),
    Mutation("groupdigits", "one character too few of a row group hashed (trailing length 5)",
             [(KERNEL, "case 3: hashRowDigits<FixedSize, 3, Inserted>(group, shape, group1, group2, sKey, sOrd); break;",
               "case 3: hashRowDigits<FixedSize, 2, Inserted>(group, shape, group1, group2, sKey, sOrd); break;")]),
    # The alphabet's size, taken at runtime (for every size but those compiled
    # in, see CompiledAlphabetSizes).
    Mutation("fastdivshift", "every runtime division by the alphabet's size or chunks per group shifted one too far",
             [(KERNEL, "return FastDivisor{(uint32_t) multiplier, shift};", "return FastDivisor{(uint32_t) multiplier, shift + 1};")]),
    Mutation("compiledall", "every alphabet searched with the first compiled-in size",
             [(KERNEL, "((shape.size == (uint32_t) Sizes ?", "((true ?")]),
    Mutation("hitk", "a hit's last character left out of its trailing index",
             [(KERNEL, "verifier_.addHits({hit.row * alphabetSize_ + hit.k},", "verifier_.addHits({hit.row * alphabetSize_},")]),
    Mutation("hostsplit", "a batch's first row put in the group after its own",
             [(KERNEL, "batch.firstGroup = (uint32_t) (firstRow / alphabetSize);", "batch.firstGroup = (uint32_t) (firstRow / alphabetSize) + 1;")]),
    Mutation("roword", "the wrong character in a row's own hash step",
             [(KERNEL, "seed2 = sOrd[d] + seed1 + group2Term;", "seed2 = sOrd[dBegin] + seed1 + group2Term;")]),
    Mutation("chunkcount", "a batch's last chunk never searched",
             [(KERNEL, "(batch.lastGroup - batch.firstGroup + 1) * chunks;", "(batch.lastGroup - batch.firstGroup + 1) * chunks - 1;")]),
    Mutation("lastgroup", "a batch's last row group never searched, when there's more than one",
             [(KERNEL, "(batch.lastGroup - batch.firstGroup + 1) * chunks;",
               "(batch.lastGroup - batch.firstGroup + (batch.lastGroup > batch.firstGroup ? 0 : 1)) * chunks;")]),
    # Several batches - usually leading values - per launch (runBatches).
    Mutation("batchseeds", "every batch of a launch searched with the first one's seeds and edges",
             [(KERNEL, "const LaunchBatch batch = batches.batch[batchIndex];", "const LaunchBatch batch = batches.batch[0];")]),
    Mutation("hitbatch", "every hit recorded as the launch's first batch's",
             [(KERNEL, "Hit{groupStart + d, batch, (uint32_t) k};", "Hit{groupStart + d, 0, (uint32_t) k};")]),
    Mutation("verifybatch", "every hit checked with the launch's first batch's prefix",
             [(KERNEL, "verifier_.addHits({hit.row * alphabetSize_ + hit.k}, trailingLen, requests[hit.batch].params, outcome);",
               "verifier_.addHits({hit.row * alphabetSize_ + hit.k}, trailingLen, requests[0].params, outcome);")]),
    # Reading the hits back.
    # Only the stress test's dense hits put more than the first few hits read
    # with the count in one launch.
    Mutation("hitsread", "the hits past the first ones read with the count read one short",
             [(KERNEL, "(hitCount - kHitsReadWithCount) * sizeof(Hit), cudaMemcpyDeviceToHost));",
               "(hitCount - kHitsReadWithCount - 1) * sizeof(Hit), cudaMemcpyDeviceToHost));")],
             caught_by=("stress",)),
    # Only a search of many launches can tell.
    Mutation("nohitreset", "the hit count never reset after a launch with hits",
             [(KERNEL, "CUDA_CHECK(cudaMemsetAsync(&bufs_.results->matchCount, 0, sizeof(int), 0));", "")],
             caught_by=("self-test", "stress-small")),
    Mutation("lastbatch", "a launch's last batch never searched, when there's more than one",
             [(KERNEL, "(unsigned) batchCount);", "(unsigned) std::max(1, batchCount - 1));")]),
    # The engine's side of it: the self-test tests the backend alone.
    Mutation("noflush", "the batches left over at the end of a candidate length never searched",
             [(ENGINE, "        if (!pending.empty()) {\n            int r = searchPending();",
               "        if (false) {\n            int r = searchPending();")],
             caught_by=("integration", "stress")),
    Mutation("groupoverflow", "batches whose hits overflowed together never searched again",
             [(ENGINE, "        for (const BatchRequest& batch : batches) {\n            int r = searchChunk(",
               "        for (const BatchRequest& batch : batches) {\n            break;\n            int r = searchChunk(")],
             caught_by=("overflow",)),
    # The kernel works out its chunks itself, so how many threads are
    # launched must not change what gets searched - whether one too few, or
    # only as many as the first batch needs.
    Mutation("hostthreads", "one thread too few launched", expect="harmless",
             edits=[(KERNEL, "const uint64_t threads = maxGroups * shape_.chunksPerGroup;",
                     "const uint64_t threads = std::max<uint64_t>(1, maxGroups * shape_.chunksPerGroup - 1);")]),
    Mutation("firstbatchgrid", "only as many threads as the first batch needs", expect="harmless",
             edits=[(KERNEL, "maxGroups = std::max(maxGroups, batchGroups);", "maxGroups = b == 0 ? batchGroups : maxGroups;")]),
    # Pruning the whole candidate: the lists of row groups, and the rows of each.
    Mutation("notlisted", "the whole candidate never pruned",
             [(KERNEL, "listed_ = constants.trailingRules.any();", "listed_ = false;")]),
    Mutation("listoffset", "every batch searching the list from the start of all of them",
             [(KERNEL, "const uint32_t* groups = bufs.groups + batch.groupsOffset;", "const uint32_t* groups = bufs.groups;")]),
    Mutation("listcount", "a batch's list's last chunk never searched",
             [(KERNEL, "const uint32_t chunkCount = batch.groupCount * chunks;", "const uint32_t chunkCount = batch.groupCount * chunks - 1;")]),
    Mutation("listgroup", "a list entry's group read with its classes' top bit",
             [(KERNEL, "const uint32_t group = entry >> kRowFlagBits;", "const uint32_t group = entry >> (kRowFlagBits - 1);")]),
    Mutation("rowflags", "every list entry's classes taken as none",
             [(KERNEL, "__ldg(&bufs.rowMasks[entry & (kRowFlagCount - 1)])", "__ldg(&bufs.rowMasks[0])")]),
    # The text inserted into the trailing part.
    Mutation("rowinsert", "the text inserted after a row's own character not hashed",
             [(KERNEL, "for (int i = 0; i < rowInsertLen; ++i)", "for (int i = 0; i < 0; ++i)")]),
    Mutation("groupinsertat", "the text inserted between a row group's characters hashed one character late",
             [(KERNEL, "if (groupDigits + 2 - d_insertLayout.groupCharsAfter[n] == at) {",
               "if (groupDigits + 1 - d_insertLayout.groupCharsAfter[n] == at) {")]),
    Mutation("rowbit", "a row pruned or not by the next row's bit",
             [(KERNEL, "ChunkRowBits rowBits = (ChunkRowBits) (rowMask >> dBegin);", "ChunkRowBits rowBits = (ChunkRowBits) (rowMask >> (dBegin + 1));")]),
    # Only a launch whose batches start in different states can tell - the
    # stress test's random ranges rarely give one.
    Mutation("rowshift", "a chunk's row bits never moved on to the next row",
             [(KERNEL, "            rowBits >>= 1;\n", "")]),
    Mutation("sliceentry", "every batch of a launch pruned as the first one",
             [(KERNEL, "rowPruning_.groupsFor(trailingLen, requests[b].params.pruneEntry,",
               "rowPruning_.groupsFor(trailingLen, requests[0].params.pruneEntry,")],
             caught_by=("self-test", "integration")),
    Mutation("stalelists", "the lists on the GPU kept when they're cleared for another alphabet",
             [(KERNEL, "if (rowPruning_.generation() != groupsGeneration_) {", "if (false) {")]),
    # A launch whose lists leave out too few row groups walks every group
    # instead, and its hits are checked against the lists on the host
    # (runBatches, RowPruning::survives).
    Mutation("unlistedhits", "the hits of a launch that didn't walk its lists all kept",
             [(KERNEL, "if (listed_ && !listed && !rowPruning_.survives(", "if (false && !rowPruning_.survives(")]),
    Mutation("survivesrow", "a hit kept whenever its row group survives, whatever its row",
             [(PRUNING, "(*entry >> kRowFlagBits) == group && (rowMasks_[*entry & (kRowFlagCount - 1)] >> d & 1);",
               "(*entry >> kRowFlagBits) == group;")]),
    # The integration test's candidates planted in a pruned row group are in
    # ranges of a few groups, which prune enough of them to walk the lists.
    Mutation("survivesgroup", "a hit kept in a pruned row group when a later one survives",
             [(PRUNING, "return entry != end && (*entry >> kRowFlagBits) == group &&", "return entry != end &&")],
             caught_by=("self-test", "stress")),
    # The self-test's pruning cases search one batch at a time.
    Mutation("survivesbatch", "every hit checked against the list of the launch's first batch",
             [(KERNEL, "!rowPruning_.survives(RowPruning::Slice{batch.groupsOffset, batch.groupCount}, hit.row)",
               "!rowPruning_.survives(RowPruning::Slice{batches.batch[0].groupsOffset, batches.batch[0].groupCount}, hit.row)")],
             caught_by=("integration", "stress")),
    # Which of the two ways a launch takes must not change what it finds.
    Mutation("alwayslisted", "every launch walking its lists", expect="harmless",
             edits=[(KERNEL, ">= groups * NAMEBREAK_LIST_MIN_PRUNED_PERCENT);", ">= 0);")]),
    Mutation("neverlisted", "no launch walking its lists", expect="harmless",
             edits=[(KERNEL, "const bool listed = listed_ && listWalking_ != ListWalking::Never &&", "const bool listed = false &&")]),
] + ROW_PRUNING_MUTATIONS + ENGINE_PRUNING_MUTATIONS

OPENCL_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(CL_KERNEL, "mask &= (1UL << ALPHABET_SIZE) - 1UL;", "mask &= 0xFFFFFFFFUL;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(CL_KERNEL, "ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];", "ulong mask = filterTable[FILTER_INDEX(seed2, seed1)];"),
              (CL_KERNEL, "? filterTable[FILTER_INDEX(seed1, seed2)] : 0UL;", "? filterTable[FILTER_INDEX(seed2, seed1)] : 0UL;")]),
    Mutation("indexshift", "the kernel's table index shifting seed2's bits one too far",
             [(CL_KERNEL, "(((seed2) & FILTER_STATE_MASK) << FILTER_BITS))", "(((seed2) & FILTER_STATE_MASK) << (FILTER_BITS + 1)))")]),
    Mutation("filterbits", "the kernel compiled for a table one bit narrower",
             [(CL_HOST, '" -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits) +', '" -DFILTER_BITS=" + std::to_string(kLowBitsFilterBits - 1) +')]),
    Mutation("lastrow", "a batch's last row one candidate short",
             [(CL_KERNEL, "mask &= (1UL << lastRowEndK) - 1UL;", "mask &= (1UL << (lastRowEndK - 1)) - 1UL;")]),
    Mutation("firstrow", "a batch's first row one candidate short",
             [(CL_KERNEL, "mask &= ~0UL << firstRowStartK;", "mask &= ~0UL << (firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(CL_KERNEL, "if (d == firstRowD)", "if (false)"), (CL_KERNEL, "if (d == lastRowD)", "if (false)")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(CL_KERNEL, "while (mask != 0) {", "while ((mask & (mask - 1UL)) != 0) {")]),
    Mutation("wrongbit", "a flagged bit taken for the character after it",
             [(CL_KERNEL, "const int k = 63 - (int) clz(mask & (0UL - mask));", "const int k = 64 - (int) clz(mask & (0UL - mask));")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(CL_KERNEL, "for (int i = 0; i < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);",
               "for (int i = 0; i + 1 < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);")]),
    WRONG_TARGET,
    # Row groups and their chunks.
    Mutation("chunkend", "every chunk's last row skipped",
             [(CL_KERNEL, "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP) - 1;")]),
    Mutation("chunkstart", "every chunk's first row skipped",
             [(CL_KERNEL, "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP) + 1;")]),
    Mutation("firstclip", "the range cut one row short at its start",
             [(CL_KERNEL, "dBegin = (int) (firstRow - groupStart);", "dBegin = (int) (firstRow - groupStart) + 1;")]),
    Mutation("lastclip", "the range cut one row short at its end",
             [(CL_KERNEL, "dEnd = (int) ((ulong) lastRow + 1 - groupStart);", "dEnd = (int) ((ulong) lastRow - groupStart);")]),
    Mutation("firstrowd", "the first row's cut applied to the wrong row",
             [(CL_KERNEL, "? (int) (firstRow % ALPHABET_SIZE) : -1;", "? (int) (firstRow % ALPHABET_SIZE) + 1 : -1;")]),
    Mutation("lastrowd", "the last row's cut applied to the wrong row",
             [(CL_KERNEL, "? (int) (lastRow % ALPHABET_SIZE) : -1;", "? (int) (lastRow % ALPHABET_SIZE) - 1 : -1;")]),
    Mutation("groupdigits", "one character too few of a row group hashed",
             [(CL_KERNEL, "for (int i = 0; i < TRAILING_LEN - 2; ++i) {", "for (int i = 0; i + 1 < TRAILING_LEN - 2; ++i) {")]),
    Mutation("roword", "the wrong character in a row's own hash step",
             [(CL_KERNEL, "MPQ_STEP(seed1, seed2, sKey[d], sOrd[d]);", "MPQ_STEP(seed1, seed2, sKey[d], sOrd[dBegin]);")]),
    Mutation("chunkcount", "a batch's last chunk never searched",
             [(CL_KERNEL, "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP - 1;")]),
    Mutation("lastgroup", "a batch's last row group never searched, when there's more than one",
             [(CL_KERNEL, "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (lastRow / ALPHABET_SIZE - firstGroup + (lastRow / ALPHABET_SIZE > firstGroup ? 0 : 1)) * CHUNKS_PER_GROUP;")]),
    # Several batches - usually leading values - per launch (runBatches).
    Mutation("batchseeds", "every batch of a launch searched with the first one's seeds and edges",
             [(CL_KERNEL, "const LaunchBatch batch = batches[batchIndex];", "const LaunchBatch batch = batches[0];")]),
    Mutation("hitbatch", "every hit recorded as the launch's first batch's",
             [(CL_KERNEL, "results->hits[slot].batch = batchIndex;", "results->hits[slot].batch = 0;")]),
    Mutation("verifybatch", "every hit checked with the launch's first batch's prefix",
             [(CL_HOST, "verifier_.addHits({hit.trailingIdx}, trailingLen, requests[hit.batch].params, outcome);",
               "verifier_.addHits({hit.trailingIdx}, trailingLen, requests[0].params, outcome);")]),
    Mutation("lastbatch", "a launch's last batch never searched, when there's more than one",
             [(CL_HOST, "(size_t) batchCount};", "(size_t) std::max(1, batchCount - 1)};")]),
    # Reading the hits back: only the stress test's dense hits put more than
    # the first few in one launch, and only a search of many launches can
    # tell a count left over from the one before.
    Mutation("hitsread", "the hits past the first ones read with the count read one short",
             [(CL_HOST, "(hitCount - kHitsReadWithCount) * sizeof(Hit), hostResults_->hits",
               "(hitCount - kHitsReadWithCount - 1) * sizeof(Hit), hostResults_->hits")],
             caught_by=("stress",)),
    Mutation("nohitreset", "the hit count never reset after a launch with hits",
             [(CL_HOST, "CL_CHECK(clEnqueueWriteBuffer(queue_, results_, CL_FALSE, offsetof(BatchResults, matchCount), sizeof(kZero), &kZero, 0, nullptr, nullptr));", "")],
             caught_by=("self-test", "stress-small")),
    # The kernel works out its chunks itself, so the global work size must
    # not change what gets searched - neither half as many work-items (but at
    # least a work-group) nor a work-group too many, nor only as many as the
    # first batch needs.
    Mutation("fewergroups", "half as many work-items launched", expect="harmless",
             edits=[(CL_HOST, "const size_t global[2] = {(chunks + local[0] - 1) / local[0] * local[0], (size_t) batchCount};",
                     "const size_t global[2] = {((chunks + 1) / 2 + local[0] - 1) / local[0] * local[0], (size_t) batchCount};")]),
    Mutation("extragroup", "a whole work-group too many launched", expect="harmless",
             edits=[(CL_HOST, "const size_t global[2] = {(chunks + local[0] - 1) / local[0] * local[0], (size_t) batchCount};",
                     "const size_t global[2] = {(chunks + 2 * local[0] - 1) / local[0] * local[0], (size_t) batchCount};")]),
    Mutation("firstbatchgrid", "only as many work-items as the first batch needs", expect="harmless",
             edits=[(CL_HOST, "maxGroups = std::max<uint64_t>(maxGroups, batch.lastRow / alphabetSize_ - batch.firstRow / alphabetSize_ + 1);",
                     "maxGroups = b == 0 ? batch.lastRow / alphabetSize_ - batch.firstRow / alphabetSize_ + 1 : maxGroups;")]),
    # Pruning the whole candidate: the lists of row groups, and the rows of each.
    Mutation("notlisted", "the whole candidate never pruned",
             [(CL_HOST, "listed_ = constants.trailingRules.any();", "listed_ = false;")]),
    Mutation("listedflag", "the kernel compiled to walk every row group, with the whole candidate pruned",
             [(CL_HOST, '" -DLISTED=" + (listed ? "1" : "0") + insertMacros_;', '" -DLISTED=0" + insertMacros_;')]),
    Mutation("listoffset", "every batch searching the list from the start of all of them",
             [(CL_KERNEL, "__global const uint* batchGroups = groups + batch.groupsOffset;", "__global const uint* batchGroups = groups;")]),
    Mutation("listcount", "a batch's list's last chunk never searched",
             [(CL_KERNEL, "(ulong) batch.groupCount * CHUNKS_PER_GROUP;", "(ulong) batch.groupCount * CHUNKS_PER_GROUP - 1;")]),
    Mutation("listgroup", "a list entry's group read with its classes' top bit",
             [(CL_KERNEL, "const uint group = entry >> ROW_FLAG_BITS;", "const uint group = entry >> (ROW_FLAG_BITS - 1);")]),
    Mutation("rowflags", "every list entry's classes taken as none",
             [(CL_KERNEL, "rowMasks[entry & (ROW_FLAG_COUNT - 1)]", "rowMasks[0]")]),
    Mutation("rowinsert", "the text inserted after a row's own character not hashed",
             [(CL_KERNEL, "            HASH_INSERTED(seed1, seed2, ROW_INSERT_START, ROW_INSERT_LEN);\n", "")]),
    Mutation("rowbit", "a row pruned or not by the next row's bit",
             [(CL_KERNEL, "ulong rowBits = rowMask >> dBegin;", "ulong rowBits = rowMask >> (dBegin + 1);")]),
    # Only a launch whose batches start in different states can tell - the
    # stress test's random ranges rarely give one.
    Mutation("rowshift", "a chunk's row bits never moved on to the next row",
             [(CL_KERNEL, "            rowBits >>= 1;\n", "")]),
    Mutation("sliceentry", "every batch of a launch pruned as the first one",
             [(CL_HOST, "rowPruning_.groupsFor(trailingLen, requests[b].params.pruneEntry,",
               "rowPruning_.groupsFor(trailingLen, requests[0].params.pruneEntry,")],
             caught_by=("self-test", "integration")),
    Mutation("stalelists", "the lists on the device kept when they're cleared for another alphabet",
             [(CL_HOST, "if (rowPruning_.generation() != groupsGeneration_) {", "if (false) {")]),
] + ROW_PRUNING_MUTATIONS

# The Metal ones can only be run on a Mac (--backend metal).
METAL_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(MTL_KERNEL, "mask &= (1ul << ALPHABET_SIZE) - 1ul;", "mask &= 0xFFFFFFFFul;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(MTL_KERNEL, "ulong mask = filterTable[FILTER_INDEX(seed1, seed2)];", "ulong mask = filterTable[FILTER_INDEX(seed2, seed1)];"),
              (MTL_KERNEL, "? filterTable[FILTER_INDEX(seed1, seed2)] : 0ul;", "? filterTable[FILTER_INDEX(seed2, seed1)] : 0ul;")]),
    Mutation("indexshift", "the kernel's table index shifting seed2's bits one too far",
             [(MTL_KERNEL, "(((seed2) & FILTER_STATE_MASK) << FILTER_BITS))", "(((seed2) & FILTER_STATE_MASK) << (FILTER_BITS + 1)))")]),
    Mutation("filterbits", "the kernel compiled for a table one bit narrower",
             [(MTL_HOST, '@"FILTER_BITS": @(kLowBitsFilterBits),', '@"FILTER_BITS": @(kLowBitsFilterBits - 1),')]),
    Mutation("lastrow", "a batch's last row one candidate short",
             [(MTL_KERNEL, "mask &= (1ul << args.lastRowEndK) - 1ul;", "mask &= (1ul << (args.lastRowEndK - 1)) - 1ul;")]),
    Mutation("firstrow", "a batch's first row one candidate short",
             [(MTL_KERNEL, "mask &= ~0ul << args.firstRowStartK;", "mask &= ~0ul << (args.firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(MTL_KERNEL, "if (d == firstRowD)", "if (false)"), (MTL_KERNEL, "if (d == lastRowD)", "if (false)")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(MTL_KERNEL, "while (mask != 0) {", "while ((mask & (mask - 1ul)) != 0) {")]),
    Mutation("wrongbit", "a flagged bit in the low half taken for the character after it",
             [(MTL_KERNEL, "(low != 0) ? int(ctz(low)) : 32 + int(ctz(high));", "(low != 0) ? int(ctz(low)) + 1 : 32 + int(ctz(high));")]),
    Mutation("highhalf", "a flagged bit in the high half taken for the character before it",
             [(MTL_KERNEL, "(low != 0) ? int(ctz(low)) : 32 + int(ctz(high));", "(low != 0) ? int(ctz(low)) : 31 + int(ctz(high));")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(MTL_KERNEL, "for (int i = 0; i < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);",
               "for (int i = 0; i + 1 < SUFFIX_LEN; ++i)\n                    MPQ_STEP(a, b, suffixKey[i], suffixOrd[i]);")]),
    WRONG_TARGET,
    # Row groups and their chunks.
    Mutation("chunkend", "every chunk's last row skipped",
             [(MTL_KERNEL, "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dEnd = (int) ((chunk + 1) * ALPHABET_SIZE / CHUNKS_PER_GROUP) - 1;")]),
    Mutation("chunkstart", "every chunk's first row skipped",
             [(MTL_KERNEL, "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP);", "int dBegin = (int) (chunk * ALPHABET_SIZE / CHUNKS_PER_GROUP) + 1;")]),
    Mutation("firstclip", "the range cut one row short at its start",
             [(MTL_KERNEL, "dBegin = (int) (args.firstRow - groupStart);", "dBegin = (int) (args.firstRow - groupStart) + 1;")]),
    Mutation("lastclip", "the range cut one row short at its end",
             [(MTL_KERNEL, "dEnd = (int) ((ulong) args.lastRow + 1 - groupStart);", "dEnd = (int) ((ulong) args.lastRow - groupStart);")]),
    Mutation("firstrowd", "the first row's cut applied to the wrong row",
             [(MTL_KERNEL, "? (int) (args.firstRow % ALPHABET_SIZE) : -1;", "? (int) (args.firstRow % ALPHABET_SIZE) + 1 : -1;")]),
    Mutation("lastrowd", "the last row's cut applied to the wrong row",
             [(MTL_KERNEL, "? (int) (args.lastRow % ALPHABET_SIZE) : -1;", "? (int) (args.lastRow % ALPHABET_SIZE) - 1 : -1;")]),
    Mutation("groupdigits", "one character too few of a row group hashed",
             [(MTL_KERNEL, "for (int i = 0; i < TRAILING_LEN - 2; ++i) {", "for (int i = 0; i + 1 < TRAILING_LEN - 2; ++i) {")]),
    Mutation("roword", "the wrong character in a row's own hash step",
             [(MTL_KERNEL, "MPQ_STEP(seed1, seed2, sKey[d], sOrd[d]);", "MPQ_STEP(seed1, seed2, sKey[d], sOrd[dBegin]);")]),
    Mutation("chunkcount", "a batch's last chunk never searched",
             [(MTL_KERNEL, "(ulong) (args.lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (args.lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP - 1;")]),
    Mutation("lastgroup", "a batch's last row group never searched, when there's more than one",
             [(MTL_KERNEL, "(ulong) (args.lastRow / ALPHABET_SIZE - firstGroup + 1) * CHUNKS_PER_GROUP;",
               "(ulong) (args.lastRow / ALPHABET_SIZE - firstGroup + (args.lastRow / ALPHABET_SIZE > firstGroup ? 0 : 1)) * CHUNKS_PER_GROUP;")]),
    # The kernel works out its chunks itself, so the grid's size must not
    # change what gets searched - neither half as many threads (but at least
    # a threadgroup) nor a threadgroup too many.
    Mutation("fewergroups", "half as many threads launched", expect="harmless",
             edits=[(MTL_HOST, "const NSUInteger threadgroups = (chunks + threadgroupSize - 1) / threadgroupSize;",
                     "const NSUInteger threadgroups = ((chunks + 1) / 2 + threadgroupSize - 1) / threadgroupSize;")]),
    Mutation("extragroup", "a whole threadgroup too many launched", expect="harmless",
             edits=[(MTL_HOST, "const NSUInteger threadgroups = (chunks + threadgroupSize - 1) / threadgroupSize;",
                     "const NSUInteger threadgroups = (chunks + threadgroupSize - 1) / threadgroupSize + 1;")]),
    # Pruning the whole candidate: the lists of row groups, and the rows of each.
    Mutation("notlisted", "the whole candidate never pruned",
             [(MTL_HOST, "listed_ = constants.trailingRules.any();", "listed_ = false;")]),
    Mutation("listedflag", "the kernel compiled to walk every row group, with the whole candidate pruned",
             [(MTL_HOST, '@"LISTED": @(listed ? 1 : 0),', '@"LISTED": @(0),')]),
    Mutation("listoffset", "the batch searching the list from the start of all of them",
             [(MTL_KERNEL, "device const uint* batchGroups = groups + args.groupsOffset;", "device const uint* batchGroups = groups;")]),
    Mutation("listcount", "the batch's list's last chunk never searched",
             [(MTL_KERNEL, "(ulong) args.groupCount * CHUNKS_PER_GROUP;", "(ulong) args.groupCount * CHUNKS_PER_GROUP - 1;")]),
    Mutation("listgroup", "a list entry's group read with its classes' top bit",
             [(MTL_KERNEL, "const uint group = entry >> ROW_FLAG_BITS;", "const uint group = entry >> (ROW_FLAG_BITS - 1);")]),
    Mutation("rowflags", "every list entry's classes taken as none",
             [(MTL_KERNEL, "rowMasks[entry & (ROW_FLAG_COUNT - 1)]", "rowMasks[0]")]),
    Mutation("rowbit", "a row pruned or not by the next row's bit",
             [(MTL_KERNEL, "ulong rowBits = rowMask >> dBegin;", "ulong rowBits = rowMask >> (dBegin + 1);")]),
    Mutation("rowshift", "a chunk's row bits never moved on to the next row",
             [(MTL_KERNEL, "            rowBits >>= 1;\n", "")]),
    Mutation("stalelists", "the lists on the GPU kept when they're cleared for another alphabet",
             [(MTL_HOST, "if (rowPruning_.generation() != groupsGeneration_) {", "if (false) {")]),
] + ROW_PRUNING_MUTATIONS

# The CPU backend: the same filter, its own row walk, and its rows shared out
# among its threads as work items.
CPU_MUTATIONS = [
    # The lookup filter's mask.
    Mutation("mask32", "a row's mask cut to 32 bits",
             [(CPU, "ctx.table[lowBitsFilterIndex(s1, s2)] & alphabetMask : 0;", "ctx.table[lowBitsFilterIndex(s1, s2)] & alphabetMask & 0xFFFFFFFFull : 0;")]),
    Mutation("swapseeds", "seed1 and seed2 swapped in the table lookup",
             [(CPU, "ctx.table[lowBitsFilterIndex(s1, s2)]", "ctx.table[lowBitsFilterIndex(s2, s1)]")]),
    Mutation("lastrow", "a batch's last row one candidate short",
             [(CPU, "mask &= (uint64_t(1) << ctx.rows.lastRowEndK) - 1;", "mask &= (uint64_t(1) << (ctx.rows.lastRowEndK - 1)) - 1;")]),
    Mutation("firstrow", "a batch's first row one candidate short",
             [(CPU, "mask &= ~uint64_t(0) << ctx.rows.firstRowStartK;", "mask &= ~uint64_t(0) << (ctx.rows.firstRowStartK + 1);")]),
    Mutation("noedges", "where the range starts and ends mid-row ignored",
             [(CPU, "if (i == 0)\n            mask &=", "if (false)\n            mask &="),
              (CPU, "if (i == ctx.rows.rowCount - 1)\n            mask &=", "if (false)\n            mask &=")]),
    Mutation("skiplast", "the loop over a row's flagged candidates stops one early",
             [(CPU, "while (mask != 0) {", "while ((mask & (mask - 1)) != 0) {")]),
    Mutation("highestbit", "a row's highest flagged bit taken for its lowest",
             [(CPU, "return __builtin_ctzll(mask);", "return 63 - __builtin_clzll(mask);")]),
    Mutation("suffixshort", "the suffix hashed one character short",
             [(CPU, "for (int j = 0; j < ctx.suffixLen; ++j)\n                mpqStep(a, b, ctx.suffixKey[j], ctx.suffixOrd[j]);",
               "for (int j = 0; j + 1 < ctx.suffixLen; ++j)\n                mpqStep(a, b, ctx.suffixKey[j], ctx.suffixOrd[j]);")]),
    WRONG_TARGET,
    # The row walk.
    Mutation("odometer", "a row's shared characters wrapping one early",
             [(CPU, "while (p >= 0 && ++digit[p] == as) {", "while (p >= 0 && ++digit[p] == as - 1) {")]),
    Mutation("rehash", "the character that changed left out of the next row's hash",
             [(CPU, "rehashFrom(p < 0 ? 0 : p);", "rehashFrom(p < 0 ? 0 : p + 1);")]),
    Mutation("startrow", "a work item's first row decoded one off",
             [(CPU, "uint64_t rowValue = ctx.rows.firstRow + from;", "uint64_t rowValue = ctx.rows.firstRow + from + 1;")]),
    # Sharing the rows out, and the batches.
    Mutation("itemgap", "a row skipped between two work items",
             [(CPU, "from < contexts[b].rows.rowCount; from += rowsPerItem)", "from < contexts[b].rows.rowCount; from += rowsPerItem + 1)")]),
    Mutation("lastbatch", "a call's last batch never searched, when there's more than one",
             [(CPU, "    for (int b = 0; b < batchCount; ++b) {\n        for (uint64_t from = 0;",
               "    for (int b = 0; b < std::max(1, batchCount - 1); ++b) {\n        for (uint64_t from = 0;")]),
    Mutation("hitbatch", "every hit recorded as the call's first batch's",
             [(CPU, "record(found, batch, row * as + k);", "record(found, 0, row * as + k);")]),
    Mutation("verifybatch", "every hit checked with the call's first batch's prefix",
             [(CPU, "requests[mine.batches[i]].params", "requests[0].params")]),
    # Only the overflow test has more hits in a call than can be recorded.
    Mutation("overflowcount", "a thread's hits counted no higher than can be recorded, hiding the overflow",
             [(CPU, "std::min<uint64_t>(mine.hitCount, MAX_MATCHES + 1)", "std::min<uint64_t>(mine.hitCount, MAX_MATCHES)")],
             caught_by=("overflow",)),
    # How the rows are shared out must not change what gets searched.
    Mutation("biggeritems", "a quarter as many work items, four times as large", expect="harmless",
             edits=[(CPU, "(totalRows + threads * 8 - 1) / (threads * 8)", "(totalRows + threads * 2 - 1) / (threads * 2)")]),
    Mutation("onethread", "every call searched on one thread", expect="harmless",
             edits=[(CPU, "    threads = std::min<uint64_t>(threads, totalRows);", "    threads = 1;")]),
    # Pruning the whole candidate: which rows the row walk searches.
    Mutation("cpunoprune", "every row searched, with the whole candidate pruned",
             [(CPU, "uint64_t mask = rowValid ? ctx.table", "uint64_t mask = true ? ctx.table")]),
    Mutation("cpuentry", "the rules started as if there were no leading characters",
             [(CPU, "pruneState[0] = ctx.pruneEntry;", "pruneState[0] = PruneState();")]),
    Mutation("cpucarry", "a row's characters checked each on their own, not after those before them",
             [(CPU, "valid[d + 1] = valid[d + 1] && pruneStep_CPU(", "valid[d + 1] = pruneStep_CPU(")]),
    Mutation("cpurowinsert", "the text inserted after a row's own character not hashed",
             [(CPU, "        stepInserted(ctx.prefixDigits, rowS1, rowS2, rowPrune, ok);\n", "")]),
] + ENGINE_PRUNING_MUTATIONS

# Per backend: how to build it, what its createBackend name is, and its mutations.
BACKENDS = {
    "cuda": (["-DNAMEBREAK_GPU=cuda", "-DNAMEBREAK_OPENCL=OFF"], CUDA_MUTATIONS),
    "hip": (["-DNAMEBREAK_GPU=hip", "-DNAMEBREAK_OPENCL=OFF"], CUDA_MUTATIONS),
    "opencl": (["-DNAMEBREAK_GPU=none", "-DNAMEBREAK_OPENCL=ON"], OPENCL_MUTATIONS),
    "metal": (["-DNAMEBREAK_GPU=none", "-DNAMEBREAK_OPENCL=OFF", "-DNAMEBREAK_METAL=ON"], METAL_MUTATIONS),
    "cpu": (["-DNAMEBREAK_GPU=none", "-DNAMEBREAK_OPENCL=OFF"], CPU_MUTATIONS),
}

CONTROL = Mutation("none", "the code as it is", [], expect="harmless")

# Runs the self-test on its own: createBackend's is switched off in the copies.
SELF_TEST_RUNNER = r'''// Written by tests/mutation_test.py into its copies of the client.
#include <cstdio>
#include <memory>
#include <string>
#include "backends/backends.h"
#include "backends/self_test.h"
int main(int argc, char** argv) {
    std::string error;
    std::unique_ptr<SearchBackend> backend = createBackend(argc > 1 ? argv[1] : "", error);
    if (!backend) {
        fprintf(stderr, "%s\n", error.c_str());
        return 77;
    }
    std::string why;
    if (selfTestBackend(*backend, why)) {
        printf("self-test passed\n");
        return 0;
    }
    printf("self-test FAILED: %s\n", why.c_str());
    return 1;
}
'''

TARGETS = ["search_integration_test", "search_stress_test", "search_overflow_test", "search_stress_test_small_launch",
           "mutation_self_test_runner"]
CHECKS = [
    ("self-test", "mutation_self_test_runner"),
    ("integration", "search_integration_test"),
    ("stress", "search_stress_test"),
    ("overflow", "search_overflow_test"),
    ("stress-small", "search_stress_test_small_launch"),
]


class ExperimentError(Exception):
    pass


@dataclass
class CheckResult:
    passed: bool
    summary: str


@dataclass
class Outcome:
    mutation: Mutation
    checks: dict = field(default_factory=dict)  # check name -> CheckResult
    error: str = ""                              # build failure etc.

    def as_expected(self):
        if self.error or len(self.checks) != len(CHECKS):
            return False
        if self.mutation.expect == "harmless":
            return all(r.passed for r in self.checks.values())
        return all(not self.checks[check].passed for check in self.mutation.caught_by)


print_lock = threading.Lock()


def say(message):
    with print_lock:
        print(message, flush=True)


def replace_exactly_once(path, old, new, what):
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise ExperimentError(f"{what}: expected to find this exactly once in {path.name}, found it {count} times:\n    {old!r}\n"
                              "The code it targets has probably changed - update the mutation in tests/mutation_test.py.")
    path.write_text(text.replace(old, new))


def prepare_base(work):
    """A copy of the client with createBackend's self-test switched off and the self-test runner added."""
    base = work / "base"
    base.mkdir()
    shutil.copy2(CLIENT / "CMakeLists.txt", base)
    for sub in ("src", "tests"):
        shutil.copytree(CLIENT / sub, base / sub, ignore=shutil.ignore_patterns("__pycache__"))
    backends = base / "src/backends/backends.cpp"
    text = backends.read_text()
    if text.count("selfTestBackend(*backend, why)") != 2:
        raise ExperimentError("couldn't find createBackend's two calls to selfTestBackend in src/backends/backends.cpp - "
                              "update prepare_base() in tests/mutation_test.py")
    backends.write_text(text.replace("selfTestBackend(*backend, why)", "true"))
    (base / "tests/mutation_self_test_runner.cpp").write_text(SELF_TEST_RUNNER)
    with open(base / "CMakeLists.txt", "a") as cmake:
        cmake.write("\n# Added by tests/mutation_test.py.\n"
                    "namebreak_test_executable(mutation_self_test_runner tests/mutation_self_test_runner.cpp namebreak_search)\n")
    return base


def build(work, base, mutation, args):
    """A copy of `base` with `mutation` applied, built. Returns its build directory."""
    src = work / mutation.name
    shutil.copytree(base, src)
    for file, old, new in mutation.edits:
        replace_exactly_once(src / file, old, new, f"mutation {mutation.name}")
    build_dir = src / "build"
    configure = ["cmake", "-S", str(src), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release", "-DNAMEBREAK_METAL=OFF",
                 "-DNAMEBREAK_NETWORK=OFF", "-DNAMEBREAK_TEST_VARIANTS=ON"] + BACKENDS[args.backend][0]
    configure += args.cmake_arg
    compile_ = ["cmake", "--build", str(build_dir), "-j", str(args.jobs), "--target"] + TARGETS
    log = src / "build.log"
    with open(log, "w") as out:
        for command in (configure, compile_):
            if subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode != 0:
                return None
    return build_dir


def summarize(check, log_text, capture_text, exit_code, timed_out):
    if timed_out:
        return "timed out"
    if check == "self-test":
        line = next((l for l in log_text.splitlines() if l.startswith("self-test")), "")
        if "INTERNAL ERROR" in log_text:
            line = log_text[log_text.index("INTERNAL ERROR"):].splitlines()[0]
        return line[:110] or f"exit {exit_code}"
    if check == "integration":
        m = re.search(r"(\d+) case\(s\) run in [\d.]+s, (\d+) failure\(s\)", log_text)
        if m:
            return f"{m.group(2)} of {m.group(1)} cases failed"
    elif check == "overflow":
        if "ALL CHECKS PASSED" in log_text:
            return "all checks passed"
        m = re.search(r"(\d+) check\(s\) FAILED", log_text)
        if m:
            return f"{m.group(1)} check(s) failed"
    else:
        m = re.search(r"(\d+) case\(s\), \d+ candidates, (\d+) hits verified, [\d.]+s, (\d+) failure\(s\)", log_text)
        if m:
            return f"{m.group(3)} of {m.group(1)} cases failed ({m.group(2)} hits verified)"
    # Stopped before its summary - e.g. by the per-search table check, whose
    # message lands in the test's output capture.
    for text in (log_text, capture_text):
        if "INTERNAL ERROR" in text:
            return "stopped: " + text[text.index("INTERNAL ERROR"):].splitlines()[0][:90]
    return f"exit {exit_code}, no summary"


def run_checks(build_dir, mutation, args):
    backend = args.backend
    results = {}
    for check, executable in CHECKS:
        run_dir = build_dir.parent / f"run-{check}"
        run_dir.mkdir(exist_ok=True)
        log_path = run_dir / "output.log"
        timed_out = False
        with open(log_path, "wb") as out:
            try:
                code = subprocess.run([str(build_dir / "tests" / executable), "--backend", backend] if check != "self-test"
                                      else [str(build_dir / "tests" / executable), backend],
                                      cwd=run_dir, stdout=out, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        if code == 77:
            raise ExperimentError(f"the {backend} backend can't run on this machine ({check}, {mutation.name}) - see {log_path}")
        log_text = log_path.read_bytes().decode("utf-8", "replace")
        capture = run_dir / ".capture.tmp"
        capture_text = capture.read_bytes().decode("utf-8", "replace") if capture.exists() else ""
        results[check] = CheckResult(passed=(code == 0), summary=summarize(check, log_text, capture_text, code, timed_out))
    return results


def describe(outcome):
    expected = outcome.mutation.expect
    if expected == "caught" and outcome.mutation.caught_by != Mutation.caught_by:
        expected += " by " + ", ".join(outcome.mutation.caught_by)
    lines = [f"{outcome.mutation.name}: {outcome.mutation.what} (expected: {expected})"]
    if outcome.error:
        lines.append(f"    ERROR: {outcome.error}")
    for check, _ in CHECKS:
        r = outcome.checks.get(check)
        if r:
            lines.append(f"    {check:<12} {'passed' if r.passed else 'FAILED':<7} {r.summary}")
    lines.append("    -> " + ("as expected" if outcome.as_expected() else "NOT AS EXPECTED"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="See the module docstring for the whole story.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="cuda", help="the backend to build and test (default cuda)")
    parser.add_argument("--only", help="comma-separated mutation names to run (the unmodified code always runs first)")
    parser.add_argument("--list", action="store_true", help="list the mutations and exit")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="parallel compile jobs per build")
    parser.add_argument("--parallel", type=int, default=3, help="mutations checked at the same time (default 3)")
    parser.add_argument("--timeout", type=int, default=900, help="seconds before one check counts as hung (default 900)")
    parser.add_argument("--work-dir", help="where to put the copies (default: a new temporary directory)")
    parser.add_argument("--keep", action="store_true", help="keep the work directory even if everything went as expected")
    parser.add_argument("--cmake-arg", action="append", default=[], help="extra CMake argument, e.g. -DCMAKE_CUDA_ARCHITECTURES=86")
    args = parser.parse_args()

    mutations = BACKENDS[args.backend][1]
    if args.list:
        for m in mutations:
            print(f"{m.name:<15} {m.expect:<9} {m.what}")
        return 0
    if args.only:
        wanted = [name.strip() for name in args.only.split(",")]
        unknown = [name for name in wanted if name not in {m.name for m in mutations}]
        if unknown:
            print(f"unknown mutation(s): {', '.join(unknown)} (see --list)", file=sys.stderr)
            return 2
        mutations = [m for m in mutations if m.name in wanted]

    work = Path(args.work_dir).resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="namebreak-mutation-"))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        print(f"the work directory {work} isn't empty", file=sys.stderr)
        return 2
    say(f"work directory: {work}")
    start = time.monotonic()
    outcomes = []
    try:
        base = prepare_base(work)
        # The unmodified code first: if it doesn't pass, nothing else means anything.
        say("building and checking the unmodified code ...")
        control = Outcome(CONTROL)
        build_dir = build(work, base, CONTROL, args)
        if build_dir is None:
            raise ExperimentError(f"the unmodified code doesn't build - see {work / CONTROL.name / 'build.log'}")
        control.checks = run_checks(build_dir, CONTROL, args)
        say(describe(control))
        if not control.as_expected():
            raise ExperimentError("the unmodified code doesn't pass all five checks, so the experiment can't tell anything")
        outcomes.append(control)

        # Each mutation is built while earlier ones are checked on the GPU.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            pending = []
            for mutation in mutations:
                say(f"building {mutation.name} ...")
                outcome = Outcome(mutation)
                build_dir = build(work, base, mutation, args)
                if build_dir is None:
                    outcome.error = f"doesn't compile - see {work / mutation.name / 'build.log'}"
                    say(describe(outcome))
                    outcomes.append(outcome)
                    continue

                def check(outcome=outcome, build_dir=build_dir):
                    outcome.checks = run_checks(build_dir, outcome.mutation, args)
                    say(describe(outcome))
                    return outcome
                pending.append(pool.submit(check))
            for future in pending:
                outcomes.append(future.result())
    except ExperimentError as e:
        print(f"\nThe experiment couldn't be run: {e}\n(work directory kept: {work})", file=sys.stderr)
        return 2

    minutes = (time.monotonic() - start) / 60
    unexpected = [o for o in outcomes if not o.as_expected()]
    widths = [max(len(c), 8) + 2 for c, _ in CHECKS]
    print(f"\n{'mutation':<15} {'expected':<9} " + "".join(f"{c:<{w}}" for (c, _), w in zip(CHECKS, widths)) + "result")
    for o in outcomes:
        # A check a mutation doesn't have to fail is in parentheses.
        cells = []
        for c, _ in CHECKS:
            cell = "-" if o.error else ("passed" if o.checks[c].passed else "failed")
            if o.mutation.expect == "caught" and c not in o.mutation.caught_by and not o.error:
                cell = f"({cell})"
            cells.append(cell)
        print(f"{o.mutation.name:<15} {o.mutation.expect:<9} " + "".join(f"{cell:<{w}}" for cell, w in zip(cells, widths)) +
              ("ok" if o.as_expected() else "NOT AS EXPECTED"))
    print(f"\n{len(outcomes)} run(s) in {minutes:.1f} minutes: {len(outcomes) - len(unexpected)} as expected, {len(unexpected)} not.")
    if unexpected:
        print("Not as expected: " + ", ".join(o.mutation.name for o in unexpected) +
              f" - a check that lets a bug through, or a mutation that isn't harmless. Work directory kept: {work}")
        return 1
    if args.keep:
        print(f"Work directory kept: {work}")
    else:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
