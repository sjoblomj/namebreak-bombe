#!/bin/bash

# namebreak reads its parameters from config.conf (see config.h) - this
# script (re)generates that file's [search] section, recomputing the resume
# point from matches.txt each time, then runs namebreak in "continuous" mode.

ALPHABET=" !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"
MAX_BACKSLASH_COUNT=0 # 0 = unlimited
PREFIX="REZ\\"
SUFFIX=".WAV"
LOWER_BOUND="REZ\\FINZ09BX.TXT"
UPPER_BOUND="REZ\\GAMEMENU.BIN"
HASH_A=0xF60F5D90
HASH_B=0xCE0A9BDB
PRUNE_SYMBOL_RUNS=true

if [ ! -f matches.txt ]; then
    start_candidate="REZ\\ .WAV"
else
    start_candidate=$(awk 'END{print}' matches.txt)
fi

# Looking for the real deal
cat > config.conf <<EOF
mode = continuous

[search]
alphabet = "$ALPHABET"
max_backslash_count = $MAX_BACKSLASH_COUNT
prefix = "$PREFIX"
suffix = "$SUFFIX"
start_candidate = "$start_candidate"
lower_bound = "$LOWER_BOUND"
upper_bound = "$UPPER_BOUND"
hash_a = $HASH_A
hash_b = $HASH_B
prune_symbol_runs = $PRUNE_SYMBOL_RUNS
EOF

./namebreak

## Looking for the real deal
# prefix = "ART\CHAT_"
# suffix = ".PCX"
# lower_bound = "ART\CHAT_BNE.PCX"
# upper_bound = "ART\CHAT____.PCX"
# hash_a = 0x888F1CE2
# hash_b = 0x447C8E70
# prune_symbol_runs = true

# Testing
# prefix = "ART\\UNIT\\OTHER\\"
# suffix = ".GRP"
# lower_bound = "ART\\UNIT\\OTHER\\ .GRP"
# upper_bound = "ART\\UNIT\\OTHER\\_.GRP"
# hash_a = 0x81C5E15F
# hash_b = 0x495816B8

# prefix = "REZ\\"
# suffix = ".BIN"
# lower_bound = "REZ\\H       .BIN"
# upper_bound = "REZ\\H_______.BIN"
# hash_a = 0x966a100f
# hash_b = 0x94926d58
