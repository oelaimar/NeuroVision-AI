#!/usr/bin/env bash
set -euo pipefail

# Get the folder containing this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Parent folder of the script
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Run everything from the parent folder
cd "$PROJECT_DIR"

mkdir -p logs

echo "== Project folder =="
echo "$PROJECT_DIR"

echo "== GPU / driver =="
nvidia-smi | tee logs/nvidia_smi.txt

echo "== Python venv =="
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -r requirements.txt

echo "== Record versions =="
{
    echo "date: $(date -u +%FT%TZ)"
    echo "python: $(python --version 2>&1)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "---- pip freeze ----"
    python -m pip freeze
} > logs/env_versions.txt

echo "== Smoke test =="
python "$SCRIPT_DIR/smoke_test_gpu.py" | tee logs/smoke_test.txt
