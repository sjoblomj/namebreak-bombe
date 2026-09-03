#!/bin/bash

if [ ! -f matches.txt ]; then
    last_match="PORTRAIT\\U \\FID00.SMK"
else
    last_match=$(awk 'END{print}' matches.txt)
fi

# Looking for the real deal
./namebreak \
  continuous \
  "$last_match" \
  "PORTRAIT\U" \
  "FID00.SMK" \
  "PORTRAIT\UKHAD\NCRTLK00.SMK" \
  "PORTRAIT\UMENGSK\UMEFID00.SMK" \
  0x63755CA3 \
  0xBCEBEE13

# Looking for PORTRAIT\PARBITER\PABFID00.SMK
#./namebreak \
#  continuous \
#  "$last_match" \
#  "PORTRAIT\P" \
#  "FID00.SMK" \
#  "PORTRAIT\PADVISOR\PADTLK00.SMK" \
#  "PORTRAIT\PARCHON\PARFID00.SMK" \
#  0xFDCCB560 \
#  0x54C5E0A2

# Looking for PORTRAIT\UFLAG2\UF2FID00.SMK
#./namebreak \
#  continuous \
#  "PORTRAIT\U FID00.SMK" \
#  "PORTRAIT\U" \
#  "FID00.SMK" \
#  "PORTRAIT\UFLAG1\UF1TLK00.SMK" \
#  "PORTRAIT\UFLAG3\UF3FID00.SMK" \
#  0x17D0F420 \
#  0xA42467DA

# Looking for PORTRAIT\UDUKE\UDUFID00.SMK
#./namebreak \
#  continuous \
#  "PORTRAIT\U FID00.SMK" \
#  "PORTRAIT\U" \
#  "FID00.SMK" \
#  "PORTRAIT\UDTEMPLAR\UDTTLK02.SMK" \
#  "PORTRAIT\UFENDRAG\UFDFID00.SMK" \
#  0xD962B57C \
#  0xC990B138

# Looking for PORTRAIT\TVESSEL\TVEFID00.SMK
#./namebreak \
#  continuous \
#  "PORTRAIT\TVESAEL\TVEFID00.SMK" \
#  "PORTRAIT\T" \
#  "FID00.SMK" \
#  "PORTRAIT\TTANK\TTATLK02.SMK" \
#  "PORTRAIT\TVULTURE\TVUFID00.SMK" \
#  0xAEB771C4 \
#  0x1162462A
