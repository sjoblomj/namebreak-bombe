#!/bin/bash

ALPHABET=" !&'()+,-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_"

if [ ! -f matches.txt ]; then
    last_match="REZ\\ .WAV"
else
    last_match=$(awk 'END{print}' matches.txt)
fi

# Looking for the real deal
./namebreak \
  continuous \
  "$ALPHABET" \
  "$last_match" \
  "REZ\\" \
  ".WAV" \
  "REZ\\FINZ09BX.TXT" \
  "REZ\\GAMEMENU.BIN" \
  0xF60F5D90 \
  0xCE0A9BDB \
  --prune-symbol-runs

## Looking for the real deal
#./namebreak \
#  continuous \
#  "$ALPHABET" \
#  "$last_match" \
#  "ART\CHAT_" \
#  ".PCX" \
#  "ART\CHAT_BNE.PCX" \
#  "ART\CHAT____.PCX" \
#  0x888F1CE2 \
#  0x447C8E70 \
#  --prune-symbol-runs

# Testing
#./namebreak \
#  continuous \
#  "$ALPHABET" \
#  "$last_match" \
#  "ART\\UNIT\\OTHER\\" \
#  ".GRP" \
#  "ART\\UNIT\\OTHER\\ .GRP" \
#  "ART\\UNIT\\OTHER\\_.GRP" \
#  0x81C5E15F \
#  0x495816B8

#./namebreak \
#  continuous \
#  "$ALPHABET" \
#  "$last_match" \
#  "REZ\\" \
#  ".BIN" \
#  "REZ\\H       .BIN" \
#  "REZ\\H_______.BIN" \
#  0x966a100f \
#  0x94926d58
