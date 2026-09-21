#ifndef NAMEBREAK_BACKENDS_BACKENDS_H
#define NAMEBREAK_BACKENDS_BACKENDS_H

#include <memory>
#include <string>
#include <vector>

#include "engine/backend.h"

// The search backends compiled into this build (see NAMEBREAK_BACKEND in CMakeLists.txt), and
// picking one of them.

// The names of the backends in this build, the default first.
std::vector<std::string> availableBackends();

// The backend called `name`, or null (with `error` set) if this build has
// none by that name.
std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error);

// The backend the program searches with: the first of availableBackends().
std::unique_ptr<SearchBackend> createDefaultBackend();

#endif // NAMEBREAK_BACKENDS_BACKENDS_H
