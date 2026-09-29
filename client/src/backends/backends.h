#ifndef NAMEBREAK_BACKENDS_BACKENDS_H
#define NAMEBREAK_BACKENDS_BACKENDS_H

#include <memory>
#include <string>
#include <vector>

#include "engine/backend.h"

// The search backends compiled into this build (see NAMEBREAK_GPU etc. in
// CMakeLists.txt), and picking one of them.

// The names of the backends in this build, most preferred first: GPU
// backends, then `cpu`, then `reference` (the slow one the others are tested
// against, which is always there).
std::vector<std::string> backendNames();

// The backend called `name` - or, if `name` is empty, the first of
// backendNames() that can run on this machine. Null, with `error` set, if
// this build has no backend by that name, or it can't run here (e.g. a GPU
// backend on a machine without that kind of GPU).
//
// Every backend it returns has just passed selfTestBackend (self_test.h) on
// this machine. One that fails it is never returned: with `name` empty, the
// next backend is tried instead, after a message on stderr; with `name` given,
// the result is null and `*selfTestFailed` (if given) is set - so a test can
// tell a backend that searches wrongly from one that just can't run here.
std::unique_ptr<SearchBackend> createBackend(const std::string& name, std::string& error, bool* selfTestFailed = nullptr);

#endif // NAMEBREAK_BACKENDS_BACKENDS_H
