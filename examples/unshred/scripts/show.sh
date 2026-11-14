#!/bin/sh
# Open the real-paper page for a saved run: scripts/show.sh [NAME]
# NAME is the photo's file name without extension (default: image).
cd "$(dirname "$0")/.." || exit 1
open "file://$PWD/web/tear.html?run=${1:-image}"
