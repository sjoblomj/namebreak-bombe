#include "common/platform.h"

#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <csignal>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // GetComputerNameA, GetUserNameA - link against Advapi32 for the latter
#include <io.h>      // _isatty, _fileno, _write
#include <conio.h>   // _getch
#else
#include <pwd.h>     // getpwuid
#include <termios.h> // tcgetattr, tcsetattr
#include <unistd.h>  // gethostname, isatty, fileno, geteuid, read, write
#endif

std::string resolveHostname() {
    char buf[256];
#ifdef _WIN32
    DWORD size = sizeof(buf);
    if (GetComputerNameA(buf, &size) && buf[0] != '\0')
        return std::string(buf);
#else
    if (gethostname(buf, sizeof(buf)) == 0) {
        buf[sizeof(buf) - 1] = '\0';
        if (buf[0] != '\0') return std::string(buf);
    }
#endif
    return "unknown-host";
}

// getpwuid(geteuid())/GetUserNameA are authoritative (don't depend on a
// shell having set an env var, and work the same whether or not there's a
// controlling terminal), so they're tried first; the env vars are only a
// fallback for the unusual case where that OS-level lookup itself fails.
std::string resolveUsername() {
#ifdef _WIN32
    char buf[256];
    DWORD size = sizeof(buf);
    if (GetUserNameA(buf, &size) && buf[0] != '\0')
        return std::string(buf);
#else
    if (struct passwd* pw = getpwuid(geteuid())) {
        if (pw->pw_name && pw->pw_name[0] != '\0')
            return pw->pw_name;
    }
#endif
    if (const char* env = std::getenv("USER");     env && env[0] != '\0') return env;
    if (const char* env = std::getenv("LOGNAME");  env && env[0] != '\0') return env;
    if (const char* env = std::getenv("USERNAME"); env && env[0] != '\0') return env; // Windows' equivalent of $USER
    return "unknown-user";
}

bool isInteractiveTerminal() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

void writeStdoutSignalSafe(const char* msg, std::size_t len) {
#ifdef _WIN32
    (void) _write(_fileno(stdout), msg, (unsigned) len);
#else
    ssize_t written = write(STDOUT_FILENO, msg, len);
    (void) written; // nothing useful to do with a failure from inside a signal handler
#endif
}

#ifdef _WIN32

bool enableRawKeypressMode() { return true; } // _getch() needs no mode change - see platform.h
void restoreKeypressMode() {}

int readKeypressBlocking() {
    for (;;) {
        int c = _getch();
        if (c == EOF)
            return -1;
        // _getch() clears ENABLE_PROCESSED_INPUT while it waits, so a Ctrl+C
        // pressed then - nearly always, with the key listener blocked in
        // here - arrives as an ordinary key (3) instead of as a SIGINT.
        // Deliver it as the SIGINT it would otherwise have been.
        if (c == 3) {
            std::raise(SIGINT);
            continue;
        }
        return c;
    }
}

#else

namespace {
struct termios g_savedTermios;
bool g_rawModeActive = false;

void restoreKeypressModeAndReraise(int sig) {
    restoreKeypressMode();
    // Put this signal's disposition back to the default (terminate) and
    // re-deliver it, rather than calling exit()/_exit() ourselves - so the
    // process still ends exactly the way it would have without this handler
    // (same exit status, same "killed by signal" semantics for a parent
    // shell/process-supervisor), just with the terminal already restored.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}
} // namespace

bool enableRawKeypressMode() {
    if (!isatty(fileno(stdin)))
        return false;
    if (tcgetattr(fileno(stdin), &g_savedTermios) != 0)
        return false;

    struct termios raw = g_savedTermios;
    raw.c_lflag &= ~(ICANON | ECHO); // read single keypresses, don't echo them - but ISIG stays on (see platform.h)
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(fileno(stdin), TCSANOW, &raw) != 0)
        return false;

    g_rawModeActive = true;
    std::signal(SIGINT, restoreKeypressModeAndReraise);
    std::signal(SIGTERM, restoreKeypressModeAndReraise);
    return true;
}

void restoreKeypressMode() {
    if (g_rawModeActive) {
        tcsetattr(fileno(stdin), TCSANOW, &g_savedTermios);
        g_rawModeActive = false;
    }
}

int readKeypressBlocking() {
    unsigned char c;
    for (;;) {
        ssize_t n = read(fileno(stdin), &c, 1);
        if (n == 1)
            return (int) c;
        // A signal (cuda_backend.cu's own SIGINT handler, e.g.) interrupting
        // this blocking read looks identical to any other error unless
        // EINTR is checked for and retried - without this, the first Ctrl+C
        // would silently kill this listener thread (read() returning -1
        // reads as "stdin closed" below) instead of leaving it to keep
        // listening for a possible second Ctrl+C or a 'p' resume.
        if (n < 0 && errno == EINTR)
            continue;
        return -1;
    }
}

#endif
