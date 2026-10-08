# cmake -DFILE=<path> [-DEXPECTED=<line>] -P check_basenames_file.cmake
# With EXPECTED: FILE has to hold exactly that one line (LF or CRLF).
# Without: FILE mustn't exist. For the dictionary-cli tests' basenames file.
if(DEFINED EXPECTED)
    if(NOT EXISTS "${FILE}")
        message(FATAL_ERROR "${FILE} wasn't written")
    endif()
    file(STRINGS "${FILE}" lines)
    if(NOT lines STREQUAL "${EXPECTED}")
        message(FATAL_ERROR "${FILE} has '${lines}', not '${EXPECTED}'")
    endif()
elseif(EXISTS "${FILE}")
    message(FATAL_ERROR "${FILE} was written, with record_basenames off")
endif()
