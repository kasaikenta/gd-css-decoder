#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++-14}
EIGEN1="/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
EIGEN2="/usr/local/include/eigen3"
EIGEN3="/usr/include/eigen3"

INC=""
for d in "$EIGEN1" "$EIGEN2" "$EIGEN3"; do
  [[ -d "$d" ]] && INC="$INC -I$d"
done

$CXX -w -std=c++23 src/gd_css_patched.cc -o gd_css -O3 -lm $INC -D_Alignof=alignof
echo "✅ Build complete: ./gd_css"
