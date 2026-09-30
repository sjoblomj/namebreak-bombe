#ifndef NAMEBREAK_BACKENDS_COMMON_LAUNCH_WAITER_H
#define NAMEBREAK_BACKENDS_COMMON_LAUNCH_WAITER_H

#include <chrono>
#include <cstdint>
#include <functional>

// Waits for a GPU launch to finish without keeping a CPU core busy for all of
// it. A GPU driver's own wait (cudaDeviceSynchronize, a blocking OpenCL read)
// spins: the client kept a core at 100% for as long as it searched, and on a
// laptop, where CPU and GPU share one power budget, that core costs the GPU
// clock. Sleeping in the driver instead (CUDA's
// cudaDeviceScheduleBlockingSync) frees it, but the thread then wakes about
// half a millisecond after the launch is done, with the GPU idle meanwhile
// (measured: the search 3.6% slower). So this sleeps through most of the
// launch, and lets the driver's wait spin only for the end of it: it predicts
// how long the launch takes from the time a candidate took in this search's
// recent large launches, and sleeps until a margin before that - twice as
// long before as its sleeps have recently overshot, so that it wakes before
// the launch is done. Where sleeps are coarse (Windows' default timer:
// 15.6 ms), the margin outgrows the launch and it never sleeps, spinning as
// before. It only decides when the driver's wait starts, never what's found.
//
// A launch is timed on the host, from launching it to the end of the wait:
// accurate when the wait still had some spinning to do. When it didn't - the
// launch was done before the sleep ended, and the GPU may have idled - that
// time says nothing about the launch, so the prediction is cut short
// instead.
class LaunchWaiter {
public:
    using Clock = std::chrono::steady_clock;

    // At every search's start: another search's kernel (alphabet, suffix) can
    // take less time per candidate. What it learned about sleeping stays.
    void beginSearch() { secondsPerCandidate_ = 0; }
    // Waits for a launch of `candidates` candidates, handed to the driver at
    // `launched` (and flushed to the device, where that's separate), to
    // finish: sleeps through most of it, then calls `driverWait`.
    void wait(uint64_t candidates, Clock::time_point launched, const std::function<void()>& driverWait);

private:
    double secondsPerCandidate_ = 0; // 0 until a large launch has been timed
    double sleepOvershoot_ = 0.5e-3; // how much later than asked a sleep has recently ended
};

#endif // NAMEBREAK_BACKENDS_COMMON_LAUNCH_WAITER_H
