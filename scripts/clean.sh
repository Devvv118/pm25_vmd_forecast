#!/usr/bin/env bash
# Wipe out everything the pipeline generates on a run, leaving the repo
# structure (and .gitkeep placeholders) intact. Does NOT delete fetched
# raw data (data/raw/) — re-run scripts/fetch_data.py yourself if you also
# want to re-fetch from OpenAQ.
#
# Usage: bash scripts/clean.sh

set -e
cd "$(dirname "$0")/.."

echo "Removing Python caches..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete

echo "Removing generated results..."
find outputs/results -type f ! -name ".gitkeep" -delete

echo "Removing saved model checkpoints..."
find outputs/models -type f ! -name ".gitkeep" -delete

echo "Removing generated plots..."
rm -rf outputs/plots
mkdir -p outputs/plots && touch outputs/plots/.gitkeep

echo "Done. outputs/ and __pycache__ are clean. data/raw/ was left untouched."
