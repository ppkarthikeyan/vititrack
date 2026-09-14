#!/bin/bash
# Commit and push everything git tracks (data/ images and models/ stay excluded by .gitignore)
cd "$(dirname "$0")/.."
git add -A
if git diff --cached --quiet; then echo "nothing to sync"; exit 0; fi
git commit -q -m "${1:-sync $(date '+%Y-%m-%d %H:%M')}"
git push && echo "synced"
