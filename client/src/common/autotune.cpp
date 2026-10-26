#include "common/autotune.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <utility>
#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "backends/backends.h"
#include "common/config.h"
#include "engine/candidate.h"
#include "engine/dictionary_search.h"
#include "engine/search.h"
#include "engine/wordlist.h"

GpuTuning tuneAxes(const GpuTuning& start, const std::vector<TuningAxis>& axes, double minGain, int maxRounds,
                   const std::function<double(const GpuTuning&)>& measure, const std::function<void(const std::string&)>& log) {
    GpuTuning current = start;
    auto say = [&](const std::string& line) {
        if (log)
            log(line);
    };
    for (int round = 0; round < maxRounds; ++round) {
        bool changed = false;
        for (const TuningAxis& axis : axes) {
            std::optional<int> GpuTuning::* field = axis.key.field;
            const int incumbent = *(current.*field);
            auto measureWith = [&](int value) {
                GpuTuning tuning = current;
                tuning.*field = value;
                const double rate = measure(tuning);
                char line[160];
                if (rate > 0)
                    snprintf(line, sizeof line, "  %s %d: %.1f G candidates/s", axis.key.name, value, rate / 1e9);
                else
                    snprintf(line, sizeof line, "  %s %d: can't be used", axis.key.name, value);
                say(line);
                return rate;
            };
            std::vector<std::pair<int, double>> rates;
            const double first = measureWith(incumbent);
            for (int value : axis.values) {
                if (value != incumbent)
                    rates.emplace_back(value, measureWith(value));
            }
            if (rates.empty())
                continue;
            const double incumbentRate = (first + measureWith(incumbent)) / 2;
            int best = incumbent;
            double bestRate = 0;
            for (const auto& [value, rate] : rates) {
                if (rate > bestRate) {
                    best = value;
                    bestRate = rate;
                }
            }
            if (bestRate > incumbentRate * (1 + minGain)) {
                current.*field = best;
                changed = true;
                say("  -> " + std::string(axis.key.name) + " " + std::to_string(best));
            } else {
                say("  -> " + std::string(axis.key.name) + " stays " + std::to_string(incumbent));
            }
        }
        if (!changed)
            break;
    }
    return current;
}

namespace {

// Sends stdout to the null device for as long as it lives: a search, and a
// backend's self-tests, print lines a trial would bury the tuning's own
// output under. Errors still go to stderr.
class SilencedStdout {
public:
    SilencedStdout() {
        fflush(stdout);
#ifdef _WIN32
        saved_ = _dup(_fileno(stdout));
        FILE* null = fopen("NUL", "w");
        if (saved_ >= 0 && null) {
            _dup2(_fileno(null), _fileno(stdout));
        }
#else
        saved_ = dup(fileno(stdout));
        FILE* null = fopen("/dev/null", "w");
        if (saved_ >= 0 && null) {
            dup2(fileno(null), fileno(stdout));
        }
#endif
        if (null)
            fclose(null);
    }
    ~SilencedStdout() {
        fflush(stdout);
        if (saved_ < 0)
            return;
#ifdef _WIN32
        _dup2(saved_, _fileno(stdout));
        _close(saved_);
#else
        dup2(saved_, fileno(stdout));
        close(saved_);
#endif
    }
    SilencedStdout(const SilencedStdout&) = delete;
    SilencedStdout& operator=(const SilencedStdout&) = delete;

private:
    int saved_ = -1;
};

using Clock = std::chrono::steady_clock;

// search_bench's search (tests/search_bench.cpp): the real 49 characters, ten
// of them, after REZ\ and before .WAV - here with nothing pruned, so that
// every candidate reaches the backend whatever its window - and a target no
// candidate of the range has. About `candidates` of them.
const std::string kRowAlphabet = " !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_";
constexpr int kRowCandidateLen = 10;

SearchRequest rowRequest(uint64_t candidates, const std::filesystem::path& dir) {
    SearchRequest req;
    req.alphabet = kRowAlphabet;
    req.prefix = "REZ\\";
    req.suffix = ".WAV";
    req.lowerBound = std::string(kRowCandidateLen, kRowAlphabet[0]);
    req.startCandidate = req.lowerBound;
    req.upperBound = indexToString(std::max<uint64_t>(candidates, 2), kRowCandidateLen, kRowAlphabet);
    req.targetHashA = 0;
    req.targetHashB = 0;
    req.outputFilePath = (dir / "matches.txt").string();
    return req;
}

// dictionary_bench's search (tests/dictionary_bench.cpp): english-1, four
// separators, after MUSIC\BG and before .WAV, with an encryption key - here
// up to three words a candidate, its first `candidates` only.
DictionaryRequest dictionaryRequest(const std::vector<std::string>& words, uint64_t candidates, const std::filesystem::path& dir) {
    DictionaryRequest req;
    req.pattern.words = words;
    req.pattern.separators = {"", "_", "-", " "};
    req.pattern.minWords = 1;
    req.pattern.maxWords = 3;
    req.endNumber = std::max<uint64_t>(candidates, 1);
    req.prefix = "MUSIC\\BG";
    req.suffix = ".WAV";
    req.targetHashA = 0x216A81D3;
    req.targetHashB = 0;
    req.checkBasename = true;
    req.basenameKey = 0x1D5AD26C;
    req.wordSource = "english-1";
    req.outputFilePath = (dir / "matches.txt").string();
    req.basenamesFilePath = (dir / "basenames.txt").string();
    req.progressFilePath = (dir / "wordnumber.txt").string();
    req.progressInterval = std::chrono::hours(1);
    req.statusInterval = std::chrono::hours(1);
    return req;
}

// One kind of search to tune for: how to time it at a size, with the backend
// made with some tuning.
struct Benchmark {
    const char* what;
    // Candidates searched per second, or 0 (with `error` set, unless it was
    // aborted) if the backend can't be made, fails its self-tests or the
    // search fails.
    std::function<double(const GpuTuning&, uint64_t candidates, std::string& error)> rate;
    uint64_t firstSize;
};

// The backend called `backend`, made with `tuning`, which createBackend
// (or the like, `create`) self-tests as it always does.
std::unique_ptr<SearchBackend> makeTuned(const std::string& backend, const GpuTuning& tuning,
                                         std::unique_ptr<SearchBackend> (*create)(const std::string&, std::string&, bool*), std::string& error) {
    BackendTunings tunings = backendTunings();
    (backend == "metal" ? tunings.metal : tunings.opencl) = tuning;
    setBackendTunings(tunings);
    SilencedStdout silenced;
    return create(backend, error, nullptr);
}

// About how many candidates take `seconds` to search with `tuning`: timed at
// growing sizes, from `benchmark.firstSize`, until one takes long enough to
// time.
uint64_t sizeFor(const Benchmark& benchmark, const GpuTuning& tuning, double seconds, std::atomic<bool>* abortRequested, std::string& error) {
    uint64_t size = benchmark.firstSize;
    for (int attempt = 0; attempt < 8; ++attempt) {
        if (abortRequested && *abortRequested)
            return 0;
        const double rate = benchmark.rate(tuning, size, error);
        if (rate <= 0)
            return 0;
        const double took = (double) size / rate;
        // Well within a row search's 49^10 candidates.
        if (took >= seconds / 4 || attempt == 7)
            return std::clamp<uint64_t>((uint64_t) (rate * seconds), 1, 1'000'000'000'000'000ULL);
        size = (uint64_t) ((double) size * std::min(100.0, seconds / 2 / took));
    }
    return 0;
}

// How long the GPU searches before the trials are sized: a GPU that starts
// cool runs faster until it's warm - a laptop's three times as fast, at
// first, and still slowing down 20 s in - which would make every trial
// longer than meant, and the first ones faster than the rest. The second
// kind of search starts with the GPU warm from the first.
constexpr double kHeatUpSeconds = 30;
constexpr double kWarmHeatUpSeconds = 5;
// What a trial costs besides its timed search, about: making the backend,
// its self-tests and a warm-up search.
constexpr double kTrialOverheadSeconds = 1.5;
// How much faster than the value it stands at another must be to be taken,
// and the result than the defaults to be written - less is noise.
constexpr double kMinGain = 0.02;

std::string gigas(double rate) {
    char text[32];
    snprintf(text, sizeof text, "%.1f", rate / 1e9);
    return text;
}

} // namespace

int runAutotune(const std::string& configPath, const std::string& requestedBackend, double secondsPerTrial, std::atomic<bool>* abortRequested) {
    const std::vector<std::string> names = backendNames();
    std::string backend = requestedBackend;
    if (backend.empty()) {
        // The backend this machine's searches run on - a CUDA build with
        // OpenCL too searches with CUDA, which tuning OpenCL wouldn't speed
        // up.
        std::string why;
        std::unique_ptr<SearchBackend> searching;
        {
            SilencedStdout silenced;
            searching = createBackend("", why);
        }
        if (!searching) {
            fprintf(stderr, "%s\n", why.c_str());
            return 1;
        }
        backend = searching->name();
    }
    if (std::find(names.begin(), names.end(), backend) == names.end()) {
        fprintf(stderr, "this build has no '%s' backend\n", backend.c_str());
        return 1;
    }
    const bool hasOpenCl = std::find(names.begin(), names.end(), "opencl") != names.end();
    if (backend == "cuda" || backend == "hip") {
        fprintf(stderr,
                "%s searches with the %s backend, whose launch settings are compiled into the program, so --autotune\n"
                "can't change them. A script in the source code builds the client with each setting and keeps the\n"
                "fastest: client/scripts/autotune/ at https://github.com/sjoblomj/namebreak-bombe - its README.md\n"
                "says what it needs.%s\n",
                requestedBackend.empty() ? "This machine" : "The client", backend.c_str(),
                hasOpenCl ? "\n(--autotune --backend opencl tunes this build's OpenCL backend instead, for searches run with --backend opencl.)"
                          : "");
        return 1;
    }
    if (gpuTuningKeys(backend).empty()) {
        fprintf(stderr, "%s searches with the %s backend, which has nothing to tune - --autotune tunes the opencl and metal backends\n",
                requestedBackend.empty() ? "This machine" : "The client", backend.c_str());
        return 1;
    }

    // The config's own [opencl]/[metal] values are what's being replaced:
    // every trial starts from the backend's defaults.
    const GpuTuning defaults = defaultGpuTuning(backend);
    auto keyNamed = [&](const char* name) {
        for (const GpuTuningKey& key : gpuTuningKeys(backend)) {
            if (std::string(key.name) == name)
                return key;
        }
        return GpuTuningKey{nullptr, nullptr, 0, 0};
    };
    auto axis = [&](const char* name, std::vector<int> values) {
        const GpuTuningKey key = keyNamed(name);
        // The default is always among the values tried.
        if (std::find(values.begin(), values.end(), *(defaults.*key.field)) == values.end())
            values.push_back(*(defaults.*key.field));
        std::sort(values.begin(), values.end());
        return TuningAxis{key, values};
    };
    // 13 rows a work-item is 4 chunks of a 49-character row group, 25 two,
    // 64 one whatever the alphabet.
    // Metal launches one batch at a time.
    const std::vector<TuningAxis> rowAxes =
        backend == "opencl" ? std::vector<TuningAxis>{axis("rows_per_thread", {7, 13, 25, 64}), axis("batches_per_launch", {4, 8, 16, 32}),
                                                      axis("window_chars", {4, 5, 6}), axis("work_group_size", {64, 128, 256, 512})}
                            : std::vector<TuningAxis>{axis("rows_per_thread", {7, 13, 25, 64}), axis("window_chars", {4, 5, 6}),
                                                      axis("work_group_size", {64, 128, 256, 512})};
    const std::vector<TuningAxis> dictionaryAxes = {axis("dictionary_words_per_thread", {8, 16, 32, 64, 128}),
                                                    axis("dictionary_work_group_size", {64, 128, 256, 512})};

    std::error_code ec;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / ("namebreak-autotune-" + std::to_string(std::random_device{}()));
    if (ec || !std::filesystem::create_directories(dir, ec)) {
        fprintf(stderr, "--autotune: can't make a temporary directory: %s\n", ec.message().c_str());
        return 1;
    }
    struct RemoveDir {
        std::filesystem::path dir;
        ~RemoveDir() {
            std::error_code ignored;
            std::filesystem::remove_all(dir, ignored);
        }
    } removeDir{dir};

    const BackendTunings configured = backendTunings();
    const Benchmark rowBenchmark = {
        "row search",
        [&](const GpuTuning& tuning, uint64_t candidates, std::string& error) -> double {
            if (abortRequested && *abortRequested)
                return 0;
            std::unique_ptr<SearchBackend> made = makeTuned(backend, tuning, createBackend, error);
            if (!made)
                return 0;
            SilencedStdout silenced;
            runSearch(*made, rowRequest(candidates / 100, dir), abortRequested);
            const Clock::time_point start = Clock::now();
            const SearchResult result = runSearch(*made, rowRequest(candidates, dir), abortRequested);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            if (!result.ok)
                error = result.error;
            return result.ok && !result.aborted ? (double) candidates / seconds : 0;
        },
        28'800'000'000ULL};
    const std::vector<std::string> words = sortedUniqueWords(builtinWordList("english-1"));
    const Benchmark dictionaryBenchmark = {
        "dictionary search",
        [&](const GpuTuning& tuning, uint64_t candidates, std::string& error) -> double {
            if (abortRequested && *abortRequested)
                return 0;
            std::unique_ptr<SearchBackend> made = makeTuned(backend, tuning, createDictionaryBackend, error);
            if (!made)
                return 0;
            SilencedStdout silenced;
            DictionaryRequest warmUp = dictionaryRequest(words, UINT64_MAX, dir);
            warmUp.pattern.words.resize(std::min<size_t>(1000, words.size()));
            warmUp.pattern.maxWords = 1;
            runDictionarySearch(*made, warmUp, abortRequested);
            std::filesystem::remove_all(dir / "wordnumber.txt", ec);
            const Clock::time_point start = Clock::now();
            const DictionaryResult result = runDictionarySearch(*made, dictionaryRequest(words, candidates, dir), abortRequested);
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            std::filesystem::remove_all(dir / "wordnumber.txt", ec);
            if (!result.ok)
                error = result.error;
            return result.ok && !result.aborted ? (double) result.candidatesSearched / seconds : 0;
        },
        1'000'000'000ULL};

    // Each axis: its values and its first value once more, twice at most;
    // then two each of defaults and result - and finding the size and
    // heating up, apart.
    int trials = 0;
    for (const std::vector<TuningAxis>* axes : {&rowAxes, &dictionaryAxes}) {
        for (const TuningAxis& tuningAxis : *axes)
            trials += 2 * ((int) tuningAxis.values.size() + 1);
        trials += 4;
    }
    {
        std::string why;
        std::unique_ptr<SearchBackend> made = makeTuned(backend, defaults, createBackend, why);
        if (!made) {
            fprintf(stderr, "%s\n", why.c_str());
            setBackendTunings(configured);
            return 1;
        }
            printf("Tuning the %s backend's launches on %s: up to %d trials, about %.0f minutes. Anything else using the\n"
               "GPU meanwhile - another namebreak, a game - makes the results meaningless. Ctrl+C stops it.\n",
               backend.c_str(), made->deviceName().c_str(), trials,
               (trials * (secondsPerTrial + kTrialOverheadSeconds) + kHeatUpSeconds + kWarmHeatUpSeconds) / 60);
    }

    std::vector<std::pair<std::string, std::string>> written;
    std::string summary;
    for (const auto& [benchmark, axes] : {std::make_pair(&rowBenchmark, &rowAxes), std::make_pair(&dictionaryBenchmark, &dictionaryAxes)}) {
        std::string error;
        printf("\n%s:\n", benchmark->what);
        fflush(stdout);
        const double heatUp = benchmark == &rowBenchmark ? kHeatUpSeconds : kWarmHeatUpSeconds;
        uint64_t size = sizeFor(*benchmark, defaults, heatUp, abortRequested, error);
        if (size != 0) {
            // A search that long heats the GPU up, and says how fast it is then.
            const double rate = benchmark->rate(defaults, size, error);
            size = rate > 0 ? std::max<uint64_t>(1, (uint64_t) (rate * secondsPerTrial)) : 0;
        }
        if (abortRequested && *abortRequested)
            break;
        if (size == 0) {
            fprintf(stderr, "the %s backend can't search with its own defaults: %s\n", backend.c_str(), error.c_str());
            setBackendTunings(configured);
            return 1;
        }
        auto measure = [&](const GpuTuning& tuning) {
            std::string why;
            const double rate = benchmark->rate(tuning, size, why);
            if (rate <= 0 && !why.empty() && !(abortRequested && *abortRequested))
                fprintf(stderr, "  (%s)\n", why.c_str());
            return rate;
        };
        const GpuTuning tuned = tuneAxes(defaults, *axes, kMinGain, 2, measure, [&](const std::string& line) {
            // Once stopped, every trial left fails at once - nothing to show.
            if (abortRequested && *abortRequested)
                return;
            printf("%s\n", line.c_str());
            fflush(stdout);
        });
        if (abortRequested && *abortRequested)
            break;
        // The defaults and the result, timed again in turn.
        double defaultRate = 0, tunedRate = 0;
        for (int i = 0; i < 2; ++i) {
            defaultRate += measure(defaults) / 2;
            tunedRate += measure(tuned) / 2;
        }
        if (abortRequested && *abortRequested)
            break;
        const bool better = tunedRate > defaultRate * (1 + kMinGain);
        printf("  defaults %s, tuned %s G candidates/s%s\n", gigas(defaultRate).c_str(), gigas(tunedRate).c_str(),
               better ? "" : " - not enough faster, keeping the defaults");
        summary += std::string(summary.empty() ? "" : "; ") + benchmark->what + " " + gigas(defaultRate) + " -> " +
                   gigas(better ? tunedRate : defaultRate) + " G candidates/s";
        for (const TuningAxis& tuningAxis : *axes) {
            const int value = *((better ? tuned : defaults).*tuningAxis.key.field);
            written.emplace_back(tuningAxis.key.name, value == *(defaults.*tuningAxis.key.field) ? "" : std::to_string(value));
        }
    }
    setBackendTunings(configured);
    if (abortRequested && *abortRequested) {
        printf("\nTuning stopped - %s left as it was.\n", configPath.c_str());
        return 1;
    }

    std::string error;
    if (!updateConfigSection(configPath, backend, written, error)) {
        fprintf(stderr, "--autotune: couldn't write %s: %s\n", configPath.c_str(), error.c_str());
        return 1;
    }
    std::string values;
    for (const auto& [key, value] : written) {
        if (!value.empty())
            values += "\n    " + key + " = " + value;
    }
    printf("\n%s\n", summary.c_str());
    if (values.empty())
        printf("The defaults were fastest: %s's [%s] has none of the tuned keys now.\n", configPath.c_str(), backend.c_str());
    else
        printf("Wrote to %s's [%s]:%s\n", configPath.c_str(), backend.c_str(), values.c_str());
    return 0;
}
