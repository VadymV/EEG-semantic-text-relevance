#!/bin/bash
set -e

echo "Starting sentence relevance benchmark..."
uv run python benchmark.py --project_path ./data --benchmark s

echo "Waiting 1 hour before starting word relevance benchmark..."
sleep 3600

echo "Starting word relevance benchmark..."
uv run python benchmark.py --project_path ./data --benchmark w

echo "Done."
