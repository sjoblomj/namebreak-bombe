# namebreak's build. The usual way in is through the presets (CMakePresets.json):
#
#   cmake --preset default            # configure into build/
#   cmake --build --preset default    # build/namebreak, and the tests
#   ctest --preset default            # run the tests
#
# See README.md's "Compiling" for the options below.
cmake_minimum_required(VERSION 3.24)
project(namebreak LANGUAGES CXX)

# Which backend the program searches with (see src/backends/ and README.md's
# "Backends"): `cuda` (the default - needs the CUDA Toolkit), or `cpu` - no
# GPU code at all, searching with the CPU reference backend every build has
# for the tests. That one is far slower; it's for checking the rest builds and
# works without CUDA.
set(NAMEBREAK_BACKEND cuda CACHE STRING "Backend the program searches with: cuda or cpu")
set_property(CACHE NAMEBREAK_BACKEND PROPERTY STRINGS cuda cpu)
if(NOT NAMEBREAK_BACKEND MATCHES "^(cuda|cpu)$")
    message(FATAL_ERROR "NAMEBREAK_BACKEND must be cuda or cpu, not '${NAMEBREAK_BACKEND}'")
endif()

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

if(NAMEBREAK_BACKEND STREQUAL "cuda")
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
endif()

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
    src/engine/mpq_hash.cpp
    src/engine/candidate.cpp
    src/backends/backends.cpp
    src/backends/cpu/cpu_backend.cpp)
if(NAMEBREAK_BACKEND STREQUAL "cuda")
    list(APPEND SEARCH_SOURCES src/backends/cuda/cuda_backend.cu)
endif()

# namebreak_search_library(<name> [<compile definition>...])
function(namebreak_search_library name)
    add_library(${name} STATIC ${SEARCH_SOURCES})
    target_include_directories(${name} PUBLIC src)
    # Public, so what's compiled against the library (the tests) sees the same values.
    target_compile_definitions(${name} PUBLIC ${ARGN})
    if(NAMEBREAK_BACKEND STREQUAL "cuda")
        target_compile_definitions(${name} PRIVATE NAMEBREAK_WITH_CUDA)
    endif()
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
    src/common/platform.cpp)
target_link_libraries(namebreak_common PUBLIC namebreak_search)

if(NAMEBREAK_NETWORK)
    add_library(namebreak_net STATIC
        src/net/coordinator_runner.cpp
        src/net/coordinator_client.cpp
        src/net/http_client.cpp
        src/net/protocol.cpp)
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
    target_link_libraries(namebreak-gui PRIVATE namebreak_net comctl32 shell32)
    if(MSVC)
        target_link_options(namebreak-gui PRIVATE /ENTRY:mainCRTStartup)
    endif()
endif()

# --- Tests and benchmarks ----------------------------------------------------

enable_testing()

# Every GPU test runs the real runSearch() - it calls fopen("matches.txt", "a")
# relative to the current directory - so each runs in its own directory under
# build/testrun/, away from any real matches and from each other (ctest -j).
# namebreak_test(<name> <executable> [<argument>...])
function(namebreak_test name executable)
    set(dir ${CMAKE_BINARY_DIR}/testrun/${name})
    file(MAKE_DIRECTORY ${dir})
    add_test(NAME ${name} COMMAND ${executable} ${ARGN} WORKING_DIRECTORY ${dir})
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

# End-to-end correctness check for runSearch() itself, against a brute-force
# reference - see the file. The CUDA backend is tested in several
# configurations of its tuning values (src/backends/cuda/tuning.h), because
# they change which code paths a search exercises - every case derives its
# geometry from them:
#   default        the shipped configuration
#   small_launch   3 rows per launch: forces many launches per leading value,
#                  so the chunk boundaries are actually exercised
#   window2        a 2-character GPU window (5 rows per launch): a much longer
#                  CPU-side leading part, and a shallower row decode
#   window6        the widest supported window: 5 prefix digits per row and
#                  trailing indices beyond 32 bits
# The CPU reference backend is tested once, with the default configuration.
namebreak_test_executable(search_integration_test tests/search_integration_test.cpp namebreak_search)
namebreak_test(integration-${NAMEBREAK_BACKEND} search_integration_test --backend ${NAMEBREAK_BACKEND})
if(NAMEBREAK_BACKEND STREQUAL "cuda")
    namebreak_test(integration-cpu search_integration_test --backend cpu)
    foreach(config IN ITEMS
            "small_launch;NAMEBREAK_ROWS_PER_LAUNCH=3"
            "window2;NAMEBREAK_GPU_WINDOW_CHARS=2;NAMEBREAK_ROWS_PER_LAUNCH=5"
            "window6;NAMEBREAK_GPU_WINDOW_CHARS=6")
        list(POP_FRONT config suffix)
        namebreak_search_library(namebreak_search_${suffix} ${config})
        namebreak_test_executable(search_integration_test_${suffix} tests/search_integration_test.cpp namebreak_search_${suffix})
        namebreak_test(integration-cuda-${suffix} search_integration_test_${suffix} --backend cuda)
    endforeach()
endif()

# What runSearch() does when a batch has more hashA hits than MAX_MATCHES can
# record - built with MAX_MATCHES=1 so a single hashA collision pair overflows it.
namebreak_search_library(namebreak_search_overflow MAX_MATCHES=1)
namebreak_test_executable(search_overflow_test tests/search_overflow_test.cpp namebreak_search_overflow)
namebreak_test(overflow-${NAMEBREAK_BACKEND} search_overflow_test --backend ${NAMEBREAK_BACKEND})
if(NAMEBREAK_BACKEND STREQUAL "cuda")
    namebreak_test(overflow-cpu search_overflow_test --backend cpu)
endif()

# Benchmarks - not tests, run with `cmake --build --preset default --target
# <name>`:
#   run_search_bench         times the real runSearch() over a ~28.8B-candidate
#                            range, on the default backend (meant for GPU
#                            backends: the CPU one would take many minutes)
#   run_prefix_hasher_bench  IncrementalPrefixHasher's own speedup, CPU only
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
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/search_bench)
add_custom_target(run_search_bench COMMAND search_bench WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/testrun/search_bench USES_TERMINAL)
add_custom_target(run_prefix_hasher_bench COMMAND prefix_hasher_bench USES_TERMINAL)
