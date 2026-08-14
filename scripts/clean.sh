#!/usr/bin/env bash
# Wipe out everything the pipeline generates on a run, leaving the repo
# structure (and .gitkeep placeholders) intact.
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

echo "Done. outputs/ and __pycache__ are clean."
