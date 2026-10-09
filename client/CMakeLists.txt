# namebreak's build. The usual way in is through the presets (CMakePresets.json):
#
#   cmake --preset default            # configure into build/
#   cmake --build --preset default    # build/namebreak, and the tests
#   ctest --preset default            # run the tests
#
# See README.md's "Compiling" for the options below.
cmake_minimum_required(VERSION 3.24)
project(namebreak LANGUAGES CXX)

# Which GPU backends to build (see src/backends/ and README.md's "Backends").
# The CPU backends are in every build.
#   NAMEBREAK_GPU  `cuda` (the default - needs the CUDA Toolkit), `hip` (AMD
#                  GPUs - the same code compiled with ROCm's HIP), or `none`
set(NAMEBREAK_GPU cuda CACHE STRING "CUDA-dialect GPU backend: cuda, hip or none")
set_property(CACHE NAMEBREAK_GPU PROPERTY STRINGS cuda hip none)
if(NOT NAMEBREAK_GPU MATCHES "^(cuda|hip|none)$")
    message(FATAL_ERROR "NAMEBREAK_GPU must be cuda, hip or none, not '${NAMEBREAK_GPU}'")
endif()

# OFF skips the tests' extra tuning configurations (see "Tests and
# benchmarks" below) - they multiply the build time and only matter on a
# machine with the GPUs to run them on. The CI release builds turn them off.
option(NAMEBREAK_TEST_VARIANTS "Build the tests' extra tuning configurations" ON)

# ON makes a test whose backend can't run on this machine fail instead of
# being skipped. For a machine that has the GPUs this build is for: there a
# backend that can't run is a broken driver - say, CUDA after a laptop's
# suspend - and skipped tests would still leave ctest at "100% tests passed".
# OFF (the default) suits machines without them, like the release workflow's.
option(NAMEBREAK_REQUIRE_GPU "Fail, rather than skip, the tests of a backend that can't run here" OFF)

# OFF builds without any networking: no coordinator client, HTTP or protocol
# code, no libcurl dependency, and `coordinator` mode is unavailable at
# runtime (the bounded and continuous modes are unaffected).
option(NAMEBREAK_NETWORK "Build coordinator mode (needs libcurl)" ON)

# Unset, CMake builds with no optimization at all - by far the easiest way to
# lose most of the search's speed without any error.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# windows.h defines min/max macros that break std::min/std::max - and it gets
# pulled in indirectly (curl/curl.h includes winsock2.h), not just by the
# files that include it themselves, so define NOMINMAX for every target.
if(WIN32)
    add_compile_definitions(NOMINMAX)
endif()

if(NAMEBREAK_GPU STREQUAL "cuda")
    # The GPU architecture to compile for, as a compute capability without the
    # dot (see `nvidia-smi --query-gpu=compute_cap --format=csv`). 86 is this
    # machine's RTX 3080 Ti Laptop GPU; override for another GPU, e.g.
    # -DCMAKE_CUDA_ARCHITECTURES=75, or `native` for whatever GPU is installed.
    # Left to CMake it would pick an old default the kernels run slower on.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
elseif(NAMEBREAK_GPU STREQUAL "hip")
    # Needs ROCm (Linux) or the HIP SDK (Windows). The AMD GPU architecture(s)
    # to compile for come from CMAKE_HIP_ARCHITECTURES, e.g. gfx1030 (Radeon
    # RX 6000) or gfx1100 (RX 7000) - see `rocminfo`; unset, CMake asks ROCm
    # which GPUs are installed.
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
endif()

# The OpenCL backend: AMD, Intel and NVIDIA GPUs through their vendors'
# regular drivers. AUTO builds it if CMake finds OpenCL's headers and loader
# library (e.g. ocl-icd-opencl-dev and opencl-headers on Debian/Ubuntu).
set(NAMEBREAK_OPENCL AUTO CACHE STRING "OpenCL backend: AUTO (if OpenCL is found), ON or OFF")
set_property(CACHE NAMEBREAK_OPENCL PROPERTY STRINGS AUTO ON OFF)
set(WITH_OPENCL OFF)
if(NAMEBREAK_OPENCL STREQUAL "ON")
    find_package(OpenCL REQUIRED)
    set(WITH_OPENCL ON)
elseif(NAMEBREAK_OPENCL STREQUAL "AUTO")
    find_package(OpenCL)
    set(WITH_OPENCL ${OpenCL_FOUND})
endif()

# The Metal backend: the Mac's GPU (Apple Silicon, or an Intel Mac's). AUTO
# builds it on macOS; it needs Xcode's command line tools, nothing else.
set(NAMEBREAK_METAL AUTO CACHE STRING "Metal backend (macOS only): AUTO (on macOS), ON or OFF")
set_property(CACHE NAMEBREAK_METAL PROPERTY STRINGS AUTO ON OFF)
set(WITH_METAL OFF)
if(NAMEBREAK_METAL STREQUAL "ON" AND NOT APPLE)
    message(FATAL_ERROR "the Metal backend can only be built on macOS")
elseif(APPLE AND NOT NAMEBREAK_METAL STREQUAL "OFF")
    set(WITH_METAL ON)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()
message(STATUS "namebreak: GPU backend ${NAMEBREAK_GPU}, Metal backend ${WITH_METAL}, OpenCL backend ${WITH_OPENCL}")

find_package(Threads REQUIRED)
if(NAMEBREAK_NETWORK)
    # On Windows, point CMake at a libcurl install, e.g. vcpkg's:
    # -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
    find_package(CURL REQUIRED)
endif()

# build/namebreak, build/tests/..., instead of scattered through the build tree.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- The search: engine + backends -------------------------------------------
#
# Built once per set of compile-time tuning values: the program uses the
# default one, and the tests several (see below), because the values change
# which code paths a search exercises.
set(SEARCH_SOURCES
    src/engine/search.cpp
    src/engine/match_writer.cpp
    src/engine/mpq_hash.cpp
    src/engine/candidate.cpp
    src/engine/dictionary.cpp
    src/engine/dictionary_search.cpp
    src/engine/wordlist.cpp
    src/backends/backends.cpp
    src/backends/self_test.cpp
    src/backends/common/dictionary_batch.cpp
    src/backends/common/launch_waiter.cpp
    src/backends/common/lowbits_filter.cpp
    src/backends/common/row_batch.cpp
    src/backends/common/row_pruning.cpp
    src/backends/cpu/cpu_backend.cpp
    src/backends/reference/reference_backend.cpp)
if(NAMEBREAK_GPU STREQUAL "cuda")
    list(APPEND SEARCH_SOURCES src/backends/cuda/cuda_backend.cu)
elseif(NAMEBREAK_GPU STREQUAL "hip")
    # The CUDA backend's source, compiled as HIP (see its gpu_runtime.h).
    list(APPEND SEARCH_SOURCES src/backends/cuda/cuda_backend.cu)
    set_source_files_properties(src/backends/cuda/cuda_backend.cu PROPERTIES LANGUAGE HIP)
endif()

# embed_kernel(<kernel source file> <header to generate, under generated/> <C++ name>)
# The OpenCL and Metal backends compile their kernels at runtime, from source
# embedded in the program as a string.
function(embed_kernel file header symbol)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
    file(READ ${file} KERNEL_SOURCE)
    set(KERNEL_FILE ${file})
    set(KERNEL_SYMBOL ${symbol})
    configure_file(src/backends/common/embedded_kernel.h.in ${CMAKE_BINARY_DIR}/generated/${header} @ONLY)
endfunction()

if(WITH_METAL)
    list(APPEND SEARCH_SOURCES src/backends/metal/metal_backend.mm)
    set_source_files_properties(src/backends/metal/metal_backend.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
    embed_kernel(src/backends/metal/search.metal backends/metal/search_kernel.h kSearchKernelSource)
    embed_kernel(src/backends/metal/dictionary.metal backends/metal/dictionary_kernel.h kDictionaryKernelSource)
endif()
if(WITH_OPENCL)
    list(APPEND SEARCH_SOURCES src/backends/opencl/opencl_backend.cpp)
    embed_kernel(src/backends/opencl/search.cl backends/opencl/search_kernel.h kSearchKernelSource)
    embed_kernel(src/backends/opencl/dictionary.cl backends/opencl/dictionary_kernel.h kDictionaryKernelSource)
endif()

# embed_word_list(<word list file> <header to generate, under generated/> <C++ name>)
# A dictionary compiled into the program (src/engine/wordlist.h), as an
# array of its words. The file must have nothing but lowercase a-z words, one
# per line, so that no word needs escaping.
function(embed_word_list file header symbol)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
    file(READ ${file} WORDS)
    string(REGEX MATCH "[^a-z\n]" BAD_CHARACTER "${WORDS}")
    if(BAD_CHARACTER)
        message(FATAL_ERROR "${file} must have only lowercase a-z words, one per line")
    endif()
    string(STRIP "${WORDS}" WORDS)
    string(REPLACE "\n" "\",\n\"" WORDS_BODY "${WORDS}")
    set(WORDS_FILE ${file})
    set(WORDS_SYMBOL ${symbol})
    configure_file(src/engine/embedded_word_list.h.in ${CMAKE_BINARY_DIR}/generated/${header} @ONLY)
endfunction()
embed_word_list(data/english-1.txt engine/english_1_words.h kEnglish1Words)

# namebreak_search_library(<name> [<compile definition>...])
function(namebreak_search_library name)
    add_library(${name} STATIC ${SEARCH_SOURCES})
    target_include_directories(${name} PUBLIC src)
    # Public, so what's compiled against the library (the tests) sees the same values.
    target_compile_definitions(${name} PUBLIC ${ARGN})
    if(NAMEBREAK_GPU STREQUAL "cuda")
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_CUDA)
    elseif(NAMEBREAK_GPU STREQUAL "hip")
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_HIP NAMEBREAK_HIP)
    endif()
    if(WITH_METAL)
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_METAL)
        target_link_libraries(${name} PUBLIC "-framework Metal" "-framework Foundation")
    endif()
    if(WITH_OPENCL)
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_OPENCL)
        target_link_libraries(${name} PUBLIC OpenCL::OpenCL)
    endif()
    target_include_directories(${name} PRIVATE ${CMAKE_BINARY_DIR}/generated)
    target_link_libraries(${name} PUBLIC Threads::Threads)
endfunction()

namebreak_search_library(namebreak_search)

# --- The program -------------------------------------------------------------

# Config file and OS helpers. platform.cpp is always built: it isn't
# networking code - it holds the OS-specific terminal/keypress helpers behind
# the pause key (used in every mode), alongside the hostname/username lookups
# the coordinator client wants.
add_library(namebreak_common STATIC
    src/common/config.cpp
    src/common/matches_file.cpp
    src/common/platform.cpp
    src/common/version.cpp)
target_link_libraries(namebreak_common PUBLIC namebreak_search)

# This build's release (see src/common/version.h) - the release workflow sets
# it to the tag it's building (-DNAMEBREAK_VERSION=v2026-09-26). Only
# version.cpp sees it, so changing it rebuilds just that file.
set(NAMEBREAK_VERSION "dev" CACHE STRING "Release this build is, e.g. v2026-09-26 ('dev' for none)")
set_source_files_properties(src/common/version.cpp PROPERTIES COMPILE_DEFINITIONS "NAMEBREAK_VERSION=\"${NAMEBREAK_VERSION}\"")

if(NAMEBREAK_NETWORK)
    add_library(namebreak_net STATIC
        src/net/basename_outbox.cpp
        src/net/coordinator_runner.cpp
        src/net/coordinator_client.cpp
        src/net/http_client.cpp
        src/net/protocol.cpp
        src/net/update_check.cpp
        src/net/word_lists.cpp)
    target_link_libraries(namebreak_net PUBLIC namebreak_common CURL::libcurl)
endif()

add_executable(namebreak src/cli/main.cpp)
target_link_libraries(namebreak PRIVATE namebreak_common)
if(NAMEBREAK_NETWORK)
    target_compile_definitions(namebreak PRIVATE NAMEBREAK_WITH_NETWORK)
    target_link_libraries(namebreak PRIVATE namebreak_net)
endif()

# A little native Win32 GUI (src/gui/win32) - Windows only. Always built with
# networking, and without the console main() (src/cli/main.cpp) - the GUI has
# its own. WIN32 plus /ENTRY:mainCRTStartup gives the process no console
# while still letting the GUI use an ordinary main() instead of WinMain() -
# see main_window.cpp's header comment.
# NOTE: there's no Windows+CUDA machine to build-verify this on.
if(WIN32 AND NAMEBREAK_NETWORK)
    file(GLOB GUI_SOURCES CONFIGURE_DEPENDS src/gui/win32/*.cpp)
    add_executable(namebreak-gui WIN32 ${GUI_SOURCES})
    target_link_libraries(namebreak-gui PRIVATE namebreak_net comctl32 comdlg32 shell32)
    if(MSVC)
        target_link_options(namebreak-gui PRIVATE /ENTRY:mainCRTStartup)
    endif()
endif()

# --- Tests and benchmarks ----------------------------------------------------

enable_testing()

# Every GPU test runs the real runSearch() - it writes matches.txt (and, on a
# find, found.txt) in the current directory - so each runs in its own
# directory under build/testrun/, away from any real matches and from each
# other (ctest -j).
# namebreak_test(<name> <executable> [<argument>...])
# A test exits with 77 when its backend can't run on this machine (a GPU
# backend without that kind of GPU), which ctest then reports as skipped -
# or, with NAMEBREAK_REQUIRE_GPU, as failed.
function(namebreak_test name executable)
    set(dir ${CMAKE_BINARY_DIR}/testrun/${name})
    file(MAKE_DIRECTORY ${dir})
    add_test(NAME ${name} COMMAND ${executable} ${ARGN} WORKING_DIRECTORY ${dir})
    if(NOT NAMEBREAK_REQUIRE_GPU)
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endfunction()

# namebreak_test_executable(<name> <source> <search library>)
function(namebreak_test_executable name source library)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${library})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
endfunction()

# Pure-CPU correctness check for IncrementalPrefixHasher (src/engine/mpq_hash.h).
namebreak_test_executable(prefix_hasher_test tests/prefix_hasher_test.cpp namebreak_search)
namebreak_test(prefix_hasher_test prefix_hasher_test)

# Pure-CPU correctness check for the lookup filter the CUDA and OpenCL backends search
# with (src/backends/common/lowbits_filter.h): the property it rests on, every
# entry of many tables, planted matches, and that its table check catches a
# wrong table.
namebreak_test_executable(lowbits_filter_test tests/lowbits_filter_test.cpp namebreak_search)
namebreak_test(lowbits_filter_test lowbits_filter_test)
# The same at other table widths, built from just the two sources it needs
# rather than a whole search library each: the table is built in 8-bit lanes
# up to 8 bits and 16-bit ones above (buildLowBitsFilterTable), so 1, 8 and 10
# bits cover both at their narrowest and widest - and 6 is the CUDA backend's
# width (kCudaLowBitsFilterBits).
foreach(bits IN ITEMS 1 6 8 10)
    add_executable(lowbits_filter_test_${bits} tests/lowbits_filter_test.cpp src/backends/common/lowbits_filter.cpp src/engine/mpq_hash.cpp)
    target_include_directories(lowbits_filter_test_${bits} PRIVATE src)
    target_compile_definitions(lowbits_filter_test_${bits} PRIVATE NAMEBREAK_LOWBITS_FILTER_BITS=${bits})
    set_target_properties(lowbits_filter_test_${bits} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    namebreak_test(lowbits_filter_test-${bits}bits lowbits_filter_test_${bits})
endforeach()

# That the backend self-test createBackend runs (src/backends/self_test.h)
# catches a backend that searches wrongly - CPU backends wrapped in broken ones.
namebreak_test_executable(self_test_test tests/self_test_test.cpp namebreak_search)
namebreak_test(self_test_test self_test_test)

# The backends this build has (backends/backends.cpp); every one is tested.
# Every backend except `reference` searches in rows (the same batch geometry
# as the CUDA one) and honours the same tuning overrides.
set(ROW_BACKENDS "")
if(NOT NAMEBREAK_GPU STREQUAL "none")
    list(APPEND ROW_BACKENDS ${NAMEBREAK_GPU})
endif()
if(WITH_METAL)
    list(APPEND ROW_BACKENDS metal)
endif()
if(WITH_OPENCL)
    list(APPEND ROW_BACKENDS opencl)
endif()
list(APPEND ROW_BACKENDS cpu)

# The word lists of a dictionary search (src/engine/wordlist.h), english-1
# pinned among them. CPU only.
namebreak_test_executable(wordlist_test tests/wordlist_test.cpp namebreak_search)
namebreak_test(wordlist_test wordlist_test)

# A dictionary search's numbering, bounds, hash helpers, counting,
# fingerprint and progress file (src/engine/dictionary.h,
# dictionary_search.h), against brute force. CPU only.
namebreak_test_executable(dictionary_test tests/dictionary_test.cpp namebreak_search)
namebreak_test(dictionary_test dictionary_test)

# What the backends that search dictionaries share (src/backends/common/
# dictionary_batch.h): the word table, the suffix filters, a launch's
# segments, checking hits. CPU only.
namebreak_test_executable(dictionary_batch_test tests/dictionary_batch_test.cpp namebreak_search)
namebreak_test(dictionary_batch_test dictionary_batch_test)

# runDictionarySearch (src/engine/dictionary_search.h) end to end on every
# backend, against brute force - after the dictionary self-test (src/backends/
# self_test.h), as createDictionaryBackend runs it. As built; with few of
# hashA's and the basename hash's bits compared (so that there are hits to
# compare, and basenames that match by chance); and with also few
# candidates and batches per call to the backend (so that a search is
# split into many calls, and stopped and resumed part way), on several CPU
# threads, and the GPU backends' launches made small: a few words to a
# thread block, so that even a small word list spans several, three tails
# to a thread, so that a word's tails do too, room for one hit of each kind
# (so that every launch with a hit is searched again, with room for them
# all), and a suffix filter of one bit, which lets about half of the
# candidates through.
set(DICTIONARY_BACKENDS ${ROW_BACKENDS} reference)
namebreak_test_executable(dictionary_search_test tests/dictionary_search_test.cpp namebreak_search)
foreach(backend IN LISTS DICTIONARY_BACKENDS)
    namebreak_test(dictionary-search-${backend} dictionary_search_test --backend ${backend})
endforeach()
if(NAMEBREAK_TEST_VARIANTS)
    namebreak_search_library(namebreak_search_dictionary_dense NAMEBREAK_HASHA_MATCH_BITS=8 NAMEBREAK_BASENAME_MATCH_BITS=8)
    namebreak_test_executable(dictionary_search_test_dense tests/dictionary_search_test.cpp namebreak_search_dictionary_dense)
    namebreak_search_library(namebreak_search_dictionary_small NAMEBREAK_HASHA_MATCH_BITS=8 NAMEBREAK_BASENAME_MATCH_BITS=8
                             NAMEBREAK_DICTIONARY_CANDIDATES_PER_CALL=97
                             NAMEBREAK_DICTIONARY_BATCHES_PER_CALL=5 NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=16
                             NAMEBREAK_DICTIONARY_THREADS_PER_BLOCK=8 NAMEBREAK_DICTIONARY_WORDS_PER_THREAD=3 NAMEBREAK_DICTIONARY_TAILS_PER_CELL=3
                             NAMEBREAK_DICTIONARY_HIT_CAPACITY=1 NAMEBREAK_DICTIONARY_FILTER_BITS=1)
    namebreak_test_executable(dictionary_search_test_small tests/dictionary_search_test.cpp namebreak_search_dictionary_small)
    foreach(backend IN LISTS DICTIONARY_BACKENDS)
        namebreak_test(dictionary-search-${backend}-dense dictionary_search_test_dense --backend ${backend})
        namebreak_test(dictionary-search-${backend}-small dictionary_search_test_small --backend ${backend})
    endforeach()
endif()

# The program itself in dictionary mode (src/cli/main.cpp), on the config
# tests/data/dictionary_cli.conf.in makes: finding REZ\CRDT_LST.TXT (exit
# code 0); not finding it, with the wrong hashB and mode given with --mode;
# and a config without max_words.
# dictionary_cli_test(<name> <mode> <hashB> <max_words line> <pass regex or ""> [<argument>...])
function(dictionary_cli_test name mode hash_b max_words regex)
    set(DICTIONARY_CLI_MODE ${mode})
    set(DICTIONARY_CLI_HASH_B ${hash_b})
    set(DICTIONARY_CLI_MAX_WORDS ${max_words})
    set(DICTIONARY_CLI_WORDS ${CMAKE_SOURCE_DIR}/tests/data/dictionary_cli_words.txt)
    configure_file(tests/data/dictionary_cli.conf.in ${CMAKE_BINARY_DIR}/testrun/${name}/config.conf @ONLY)
    namebreak_test(${name} namebreak --config config.conf ${ARGN})
    if(regex)
        set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
    endif()
endfunction()
dictionary_cli_test(dictionary-cli-found dictionary 0xE6B2B01C "max_words = 2" "")
dictionary_cli_test(dictionary-cli-not-found bounded 0x00000000 "max_words = 2" "Searched every candidate - no match of both hashes"
                    --mode dictionary)
dictionary_cli_test(dictionary-cli-bad-config dictionary 0xE6B2B01C "" "missing required key 'max_words'")
dictionary_cli_test(dictionary-cli-no-basenames dictionary 0xE6B2B01C "max_words = 2\nrecord_basenames = false" "")
# The basenames file those runs leave (record_basenames, on by default with
# an encryption key): the find's basename, or - turned off - no file.
foreach(run found no-basenames)
    file(REMOVE ${CMAKE_BINARY_DIR}/testrun/dictionary-cli-${run}/matches/basenames-credits.txt)
    set_tests_properties(dictionary-cli-${run} PROPERTIES FIXTURES_SETUP dictionary-cli-${run}-ran)
endforeach()
add_test(NAME dictionary-cli-basenames-file
    COMMAND ${CMAKE_COMMAND} -DFILE=matches/basenames-credits.txt -DEXPECTED=CRDT_LST.TXT -P ${CMAKE_SOURCE_DIR}/tests/check_basenames_file.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/dictionary-cli-found)
set_tests_properties(dictionary-cli-basenames-file PROPERTIES FIXTURES_REQUIRED dictionary-cli-found-ran)
add_test(NAME dictionary-cli-no-basenames-file
    COMMAND ${CMAKE_COMMAND} -DFILE=matches/basenames-credits.txt -P ${CMAKE_SOURCE_DIR}/tests/check_basenames_file.cmake
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/dictionary-cli-no-basenames)
set_tests_properties(dictionary-cli-no-basenames-file PROPERTIES FIXTURES_REQUIRED dictionary-cli-no-basenames-ran)

# Where buildSearchRequest (src/common/config.h) starts a search -
# start_candidate and resume_from_last_candidate. CPU only.
namebreak_test_executable(config_test tests/config_test.cpp namebreak_common)
namebreak_test(config_test config_test)

# readLastLines (src/common/matches_file.h), which reads the end of a matches
# file without reading the rest of it. CPU only.
namebreak_test_executable(matches_file_test tests/matches_file_test.cpp namebreak_common)
namebreak_test(matches_file_test matches_file_test)

# Release versions (src/common/version.h) and, with networking, the pure
# parts of the startup update check. No network access.
if(NAMEBREAK_NETWORK)
    namebreak_test_executable(version_test tests/version_test.cpp namebreak_net)
    target_compile_definitions(version_test PRIVATE NAMEBREAK_WITH_NETWORK)
else()
    namebreak_test_executable(version_test tests/version_test.cpp namebreak_common)
endif()
namebreak_test(version_test version_test)

# Dictionary targets end to end: the coordinator server (../coordinator,
# built with cargo) and this client in coordinator mode, over HTTP on
# localhost - see the script. Needs cargo and Python 3; skipped without. It
# runs the program as it is, and built two other ways:
#   namebreak_heartbeat_1s             a heartbeat every second instead of
#                                      every minute, for the parts that have
#                                      the server go away and come back
#   namebreak_one_basename_per_report  at most one basename a report (rather
#                                      than 5,000), and a heartbeat every 10
#                                      seconds - for the parts with more
#                                      basenames than a report carries
if(NAMEBREAK_NETWORK)
    # namebreak_test_program(<name> <definition>...): the program, its
    # coordinator code compiled with the definitions.
    function(namebreak_test_program name)
        add_library(${name}_net STATIC
            src/net/basename_outbox.cpp
            src/net/coordinator_runner.cpp
            src/net/coordinator_client.cpp
            src/net/http_client.cpp
            src/net/protocol.cpp
            src/net/update_check.cpp
            src/net/word_lists.cpp)
        target_compile_definitions(${name}_net PRIVATE ${ARGN})
        target_link_libraries(${name}_net PUBLIC namebreak_common CURL::libcurl)
        add_executable(${name} src/cli/main.cpp)
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_NETWORK)
        target_link_libraries(${name} PRIVATE ${name}_net)
        set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    endfunction()
    namebreak_test_program(namebreak_heartbeat_1s NAMEBREAK_HEARTBEAT_INTERVAL_SECONDS=1)
    namebreak_test_program(namebreak_one_basename_per_report NAMEBREAK_HEARTBEAT_INTERVAL_SECONDS=10 NAMEBREAK_MAX_BASENAMES_PER_REPORT=1)
    find_program(CARGO_EXECUTABLE cargo)
    find_program(PYTHON3_EXECUTABLE python3)
    if(CARGO_EXECUTABLE AND PYTHON3_EXECUTABLE)
        namebreak_test(coordinator-dictionary ${PYTHON3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/coordinator_dictionary_test.py
            --client $<TARGET_FILE:namebreak> --client-heartbeat-1s $<TARGET_FILE:namebreak_heartbeat_1s>
            --client-one-basename $<TARGET_FILE:namebreak_one_basename_per_report>
            --coordinator ${CMAKE_SOURCE_DIR}/../coordinator)
        set_tests_properties(coordinator-dictionary PROPERTIES TIMEOUT 900)
    endif()
endif()

# The coordinator client's pure parts: the protocol's JSON
# (src/net/protocol.h), the basenames waiting to be sent
# (src/net/basename_outbox.h) and a dictionary claim's word lists
# (src/net/word_lists.h). No network access.
if(NAMEBREAK_NETWORK)
    foreach(test protocol_test basename_outbox_test word_lists_test)
        namebreak_test_executable(${test} tests/${test}.cpp namebreak_net)
        namebreak_test(${test} ${test})
    endforeach()
endif()

# End-to-end correctness check for runSearch() itself, against a brute-force
# reference - see the file. Every backend is tested with its shipped tuning
# values, and the row-based ones in a few more configurations
# (NAMEBREAK_GPU_WINDOW_CHARS/NAMEBREAK_ROWS_PER_LAUNCH, see
# src/backends/cuda/tuning.h), because they change which code paths a search
# exercises - every case derives its geometry from them:
#   small_launch   3 rows per launch: forces many launches per leading value,
#                  so the chunk boundaries are actually exercised
#   window2        a 2-character window (5 rows per launch): a much longer
#                  CPU-side leading part, and a shallower row decode
#   window6        the widest supported window: 5 prefix digits per row and
#                  trailing indices beyond 32 bits
# Each also splits the CUDA kernel's row groups differently
# (NAMEBREAK_ROWS_PER_THREAD, see kChunksPerGroup in cuda_backend.cu): 3 rows
# per thread (uneven chunks, cut up by the tiny launches), 1 (a thread per
# row), and 64 (a thread per whole group) - and searches its batches
# (NAMEBREAK_BATCHES_PER_LAUNCH, see runBatches in cuda_backend.cu) 3, 5 and
# 32 at a time: with batches of a few rows, a launch then often spans several
# leading values, with pruned ones between them. All three also make the cpu backend
# split every call, however small, across its threads. And a search that prunes
# the whole candidate walks its lists of row groups on CUDA/HIP and OpenCL only
# where they leave out enough of them (NAMEBREAK_LIST_MIN_PRUNED_PERCENT), so
# small_launch always walks them (0) and window2 never does (101), its hits
# checked against the lists on the host instead; the shipped default and window6
# do some of each.
namebreak_test_executable(search_integration_test tests/search_integration_test.cpp namebreak_search)
foreach(backend IN LISTS ROW_BACKENDS ITEMS reference)
    namebreak_test(integration-${backend} search_integration_test --backend ${backend})
endforeach()
if(ROW_BACKENDS AND NAMEBREAK_TEST_VARIANTS)
    foreach(config IN ITEMS
            "small_launch;NAMEBREAK_ROWS_PER_LAUNCH=3;NAMEBREAK_ROWS_PER_THREAD=3;NAMEBREAK_BATCHES_PER_LAUNCH=3;NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1;NAMEBREAK_LIST_MIN_PRUNED_PERCENT=0"
            "window2;NAMEBREAK_GPU_WINDOW_CHARS=2;NAMEBREAK_ROWS_PER_LAUNCH=5;NAMEBREAK_ROWS_PER_THREAD=1;NAMEBREAK_BATCHES_PER_LAUNCH=5;NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1;NAMEBREAK_LIST_MIN_PRUNED_PERCENT=101"
            "window6;NAMEBREAK_GPU_WINDOW_CHARS=6;NAMEBREAK_ROWS_PER_THREAD=64;NAMEBREAK_BATCHES_PER_LAUNCH=32;NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1")
        list(POP_FRONT config suffix)
        namebreak_search_library(namebreak_search_${suffix} ${config})
        namebreak_test_executable(search_integration_test_${suffix} tests/search_integration_test.cpp namebreak_search_${suffix})
        foreach(backend IN LISTS ROW_BACKENDS)
            namebreak_test(integration-${backend}-${suffix} search_integration_test_${suffix} --backend ${backend})
        endforeach()
    endforeach()
endif()

# Known-answer test: every name of a real StarCraft listfile
# (tests/data/sc-listfile.txt) must be found from its two hashes, each in a
# small search planted around it - see the file. Every backend, with its
# shipped tuning; `reference`, which hashes every candidate from scratch, on
# every 5th name only.
namebreak_test_executable(listfile_test tests/listfile_test.cpp namebreak_search)
foreach(backend IN LISTS ROW_BACKENDS)
    namebreak_test(listfile-${backend} listfile_test --backend ${backend} --listfile ${CMAKE_SOURCE_DIR}/tests/data/sc-listfile.txt)
endforeach()
namebreak_test(listfile-reference listfile_test --backend reference --listfile ${CMAKE_SOURCE_DIR}/tests/data/sc-listfile.txt --every 5)

# What runSearch() does when a batch has more hashA hits than MAX_MATCHES can
# record - built with MAX_MATCHES=1 so a single hashA collision pair overflows it.
namebreak_search_library(namebreak_search_overflow MAX_MATCHES=1)
namebreak_test_executable(search_overflow_test tests/search_overflow_test.cpp namebreak_search_overflow)
foreach(backend IN LISTS ROW_BACKENDS ITEMS reference)
    namebreak_test(overflow-${backend} search_overflow_test --backend ${backend})
endforeach()

# Dense-hit differential test: built so that a hashA "hit" only has to match
# the target's low 12 bits (NAMEBREAK_HASHA_MATCH_BITS, src/engine/hash_match.h),
# about one candidate in 4096, and compared hit for hit with a brute force
# over hundreds of random ranges - see the file. Every backend. The variants (row-based
# backends only - `reference` ignores all of these values) mirror the
# integration test's geometries, plus a lookup filter of a single bit
# (NAMEBREAK_LOWBITS_FILTER_BITS=1), which lets about half of every row
# through to be hashed in full, and small_launch's geometry with room for a
# single hit per launch (MAX_MATCHES=1), so that batches - and batches
# searched together - overflow all the time, and the engine's fallbacks of
# searching them again in halves, or one by one, run on random ranges. As in
# the integration test, small_launch always walks its lists of row groups and
# window2 never does - nor does overflow, whose launches then overflow with
# hits the host would have dropped too.
set(STRESS_BACKENDS ${ROW_BACKENDS})
set(STRESS_DEFINITIONS NAMEBREAK_HASHA_MATCH_BITS=12)
namebreak_search_library(namebreak_search_stress ${STRESS_DEFINITIONS})
namebreak_test_executable(search_stress_test tests/search_stress_test.cpp namebreak_search_stress)
foreach(backend IN LISTS STRESS_BACKENDS ITEMS reference)
    namebreak_test(stress-${backend} search_stress_test --backend ${backend})
endforeach()
if(ROW_BACKENDS AND NAMEBREAK_TEST_VARIANTS)
    set(STRESS_small_launch NAMEBREAK_ROWS_PER_LAUNCH=3 NAMEBREAK_ROWS_PER_THREAD=3 NAMEBREAK_BATCHES_PER_LAUNCH=3 NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1 NAMEBREAK_LIST_MIN_PRUNED_PERCENT=0)
    set(STRESS_window2 NAMEBREAK_GPU_WINDOW_CHARS=2 NAMEBREAK_ROWS_PER_LAUNCH=5 NAMEBREAK_ROWS_PER_THREAD=1 NAMEBREAK_BATCHES_PER_LAUNCH=5 NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1 NAMEBREAK_LIST_MIN_PRUNED_PERCENT=101)
    set(STRESS_window6 NAMEBREAK_GPU_WINDOW_CHARS=6 NAMEBREAK_ROWS_PER_THREAD=64 NAMEBREAK_BATCHES_PER_LAUNCH=32 NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1)
    set(STRESS_filter1 NAMEBREAK_LOWBITS_FILTER_BITS=1)
    set(STRESS_overflow MAX_MATCHES=1 NAMEBREAK_ROWS_PER_LAUNCH=3 NAMEBREAK_ROWS_PER_THREAD=3 NAMEBREAK_BATCHES_PER_LAUNCH=3 NAMEBREAK_CPU_MIN_CANDIDATES_PER_THREAD=1 NAMEBREAK_LIST_MIN_PRUNED_PERCENT=101)
    foreach(variant IN ITEMS small_launch window2 window6 filter1 overflow)
        namebreak_search_library(namebreak_search_stress_${variant} ${STRESS_DEFINITIONS} ${STRESS_${variant}})
        namebreak_test_executable(search_stress_test_${variant} tests/search_stress_test.cpp namebreak_search_stress_${variant})
        foreach(backend IN LISTS STRESS_BACKENDS)
            namebreak_test(stress-${backend}-${variant} search_stress_test_${variant} --backend ${backend})
        endforeach()
    endforeach()
endif()

# Benchmarks - not tests, run with `cmake --build --preset default --target
# <name>`:
#   run_search_bench         times the real runSearch() over a ~28.8B-candidate
#                            range, on the default backend (run
#                            build/tests/search_bench --backend <name> for
#                            another; not `reference`, which would take hours)
#   run_prefix_hasher_bench  IncrementalPrefixHasher's own speedup, CPU only
#   run_dictionary_bench     times a real dictionary search - two words of
#                            english-1, about 16.3 billion candidates - on the
#                            default backend (build/tests/dictionary_bench
#                            --backend <name> for another; --words <n> to
#                            time the CPU backend on fewer; --no-key to
#                            hash hashA, not the basename;
#                            --record-hasha-matches to hash both)
# NAMEBREAK_BENCH_WINDOW sets the GPU window for search_bench only, to
# re-sweep that tuning value against real hardware without hand-editing the
# source - e.g. -DNAMEBREAK_BENCH_WINDOW=6. Empty, the shipped default is used.
set(NAMEBREAK_BENCH_WINDOW "" CACHE STRING "GPU window for search_bench (empty: the shipped default)")
if(NAMEBREAK_BENCH_WINDOW)
    namebreak_search_library(namebreak_search_bench NAMEBREAK_GPU_WINDOW_CHARS=${NAMEBREAK_BENCH_WINDOW})
    namebreak_test_executable(search_bench tests/search_bench.cpp namebreak_search_bench)
else()
    namebreak_test_executable(search_bench tests/search_bench.cpp namebreak_search)
endif()
namebreak_test_executable(prefix_hasher_bench tests/prefix_hasher_bench.cpp namebreak_search)
namebreak_test_executable(dictionary_bench tests/dictionary_bench.cpp namebreak_search)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/dictionary_bench)
add_custom_target(run_dictionary_bench COMMAND dictionary_bench WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/dictionary_bench USES_TERMINAL)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/search_bench)
# A GPU covers the default range in hundredths of a second; 20 times it is long enough to time.
if(NAMEBREAK_GPU STREQUAL "none")
    set(SEARCH_BENCH_ARGS "")
else()
    set(SEARCH_BENCH_ARGS --scale 20)
endif()
add_custom_target(run_search_bench COMMAND search_bench ${SEARCH_BENCH_ARGS} WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/search_bench USES_TERMINAL)

# Mutation testing (tests/mutation_test.py) - not a ctest test: it builds
# a copy of the client per mutation, with a GPU kernel broken on purpose, and
# checks that the self-test, the integration test and the stress test (and,
# for the ones only they can see, the overflow test and the small-launch
# stress test) catch every one. About fifteen minutes for CUDA's list and
# four for OpenCL's on the RTX 3080 Ti Laptop: `cmake --build --preset
# default --target run_mutation_test` (the CUDA or HIP kernel) or
# run_mutation_test_opencl, after any change to the kernel.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND AND WITH_OPENCL)
    add_custom_target(run_mutation_test_opencl COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/mutation_test.py --backend opencl
                      USES_TERMINAL VERBATIM)
endif()
if(NOT NAMEBREAK_GPU STREQUAL "none")
    if(Python3_FOUND)
        set(MUTATION_TEST_ARGS --backend ${NAMEBREAK_GPU})
        # A single architecture is passed on; the script's copies build for
        # CMake's default otherwise.
        if(NAMEBREAK_GPU STREQUAL "cuda" AND CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES MATCHES ";")
            list(APPEND MUTATION_TEST_ARGS "--cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        elseif(NAMEBREAK_GPU STREQUAL "hip" AND CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES MATCHES ";")
            list(APPEND MUTATION_TEST_ARGS "--cmake-arg=-DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
        endif()
        add_custom_target(run_mutation_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/mutation_test.py ${MUTATION_TEST_ARGS}
                          USES_TERMINAL VERBATIM)
    endif()
endif()
add_custom_target(run_prefix_hasher_bench COMMAND prefix_hasher_bench USES_TERMINAL)
