#include "platform.h"

#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // GetComputerNameA, GetUserNameA - link against Advapi32 for the latter
#include <io.h>      // _isatty, _fileno
#else
#include <pwd.h>     // getpwuid
#include <unistd.h>  // gethostname, isatty, fileno, geteuid
#endif

std::string resolveHostname() {
    char buf[256];
#ifdef _WIN32
    DWORD size = sizeof(buf);
    if (GetComputerNameA(buf, &size) && buf[0] != '\0') return std::string(buf);
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
    if (GetUserNameA(buf, &size) && buf[0] != '\0') return std::string(buf);
#else
    if (struct passwd* pw = getpwuid(geteuid())) {
        if (pw->pw_name && pw->pw_name[0] != '\0') return pw->pw_name;
    }
#endif
    if (const char* env = std::getenv("USER"); env && env[0] != '\0') return env;
    if (const char* env = std::getenv("LOGNAME"); env && env[0] != '\0') return env;
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
