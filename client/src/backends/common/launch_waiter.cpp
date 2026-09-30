#include "backends/common/launch_waiter.h"

#include <algorithm>
#include <thread>

void LaunchWaiter::wait(uint64_t candidates, Clock::time_point launched, const std::function<void()>& driverWait) {
    // Launches too small to time well - the fixed cost of one would dominate
    // - are neither predicted from nor learned from.
    constexpr uint64_t kLearnFrom = 100'000'000;
    constexpr double kShare = 0.95;           // of the predicted time, at most
    constexpr double kMinSleep = 0.2e-3;      // shorter isn't worth a sleep
    constexpr double kSpunLongEnough = 20e-6; // a wait this long had spinning to do
    auto since = [](Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); };
    const bool large = candidates >= kLearnFrom;
    bool slept = false;
    if (large && secondsPerCandidate_ > 0) {
        const double sleep = kShare * (double) candidates * secondsPerCandidate_ - since(launched) - 2 * sleepOvershoot_;
        if (sleep >= kMinSleep) {
            const Clock::time_point before = Clock::now();
            std::this_thread::sleep_for(std::chrono::duration<double>(sleep));
            const double overshoot = since(before) - sleep;
            // Up at once, down slowly: one late wake-up is reason enough to
            // keep a wider margin for a while.
            sleepOvershoot_ = std::max(overshoot, 0.9 * sleepOvershoot_ + 0.1 * overshoot);
            slept = true;
        }
    }
    const Clock::time_point waitStart = Clock::now();
    driverWait(); // spins for the rest
    if (!large)
        return;
    if (slept && since(waitStart) < kSpunLongEnough) {
        secondsPerCandidate_ *= 0.9; // overslept: predict less next time
        return;
    }
    const double perCandidate = since(launched) / (double) candidates;
    secondsPerCandidate_ = secondsPerCandidate_ > 0 ? 0.7 * secondsPerCandidate_ + 0.3 * perCandidate : perCandidate;
}
