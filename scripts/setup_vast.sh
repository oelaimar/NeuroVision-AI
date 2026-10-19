#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "== GPU / driver =="
nvidia-smi | tee logs/nvidia_smi.txt

echo "== Python venv =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

echo "== Record versions =="
{
  echo "date: $(date -u +%FT%TZ)"
  echo "python: $(python --version 2>&1)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "---- pip freeze ----"
  pip freeze
} > logs/env_versions.txt

echo "== Smoke test =="
python scripts/smoke_test_gpu.py | tee logs/smoke_test.txt