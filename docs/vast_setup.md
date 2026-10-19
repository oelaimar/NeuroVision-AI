Configure the Vast.ai GPU instance

Prepare the Vast.ai GPU environment for repeatable training.
Acceptance criteria:

    GPU and driver are detected by the chosen framework.
    Required Python dependencies are installed and versions are recorded.
    A small smoke test can allocate the GPU and run a forward pass.
    Connection and startup instructions are documented without secrets.

give me what to do
Recalled memory

Assuming TensorFlow, since your MRI pipeline uses tf.data. Pick a Vast.ai template with a CUDA image (e.g. the TensorFlow or PyTorch template, Ubuntu 22.04), SSH enabled, and 30 GB+ disk.

1. requirements-gpu.txt

tensorflow[and-cuda]==2.17.1
numpy<2
opencv-python-headless
scikit-learn
matplotlib
pandas
jupyterlab

2. scripts/setup_vast.sh
bash

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

3. scripts/smoke_test_gpu.py
python

import sys
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.sysconfig.get_build_info().get("cuda_version"))
print("GPUs:", gpus)
if not gpus:
    sys.exit("FAIL: no GPU detected by TensorFlow")

for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

with tf.device("/GPU:0"):
    # allocation + matmul
    a = tf.random.normal((2048, 2048))
    b = tf.random.normal((2048, 2048))
    c = tf.matmul(a, b)
    print("matmul device:", c.device)

    # forward pass: small CNN, 224x224x3, 4 classes (glioma/meningioma/notumor/pituitary)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((224, 224, 3)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    x = np.random.rand(8, 224, 224, 3).astype("float32")
    y = model(x, training=False)
    print("forward output shape:", y.shape, "| device:", y.device)

assert y.shape == (8, 4)
print("SMOKE TEST PASSED")

4. docs/vast_setup.md (no secrets)
markdown

# Vast.ai GPU setup

## Rent
- Template: CUDA/TensorFlow image, Ubuntu 22.04, SSH mode
- Disk: 30 GB+, GPU: 1x RTX 3090/4090 or better
- Add your PUBLIC SSH key in Vast.ai account settings (never commit private keys or API keys)

## Connect
```bash
ssh -p <PORT> root@<HOST>          # values shown in the Vast.ai instance panel
```

## First-time setup
```bash
git clone <repo-url> && cd <repo>
bash scripts/setup_vast.sh
```
Outputs: `logs/nvidia_smi.txt`, `logs/env_versions.txt`, `logs/smoke_test.txt`.

## Every session
```bash
cd <repo> && source .venv/bin/activate
tmux new -s train                  # reattach: tmux attach -t train
python -m src.train                # your training entrypoint
```

## Copy data / results
```bash
rsync -avz -e "ssh -p <PORT>" data/raw/ root@<HOST>:~/<repo>/data/raw/
rsync -avz -e "ssh -p <PORT>" root@<HOST>:~/<repo>/outputs/ ./outputs/
```

## Jupyter (optional)
```bash
jupyter lab --no-browser --port 8888
# on your laptop:
ssh -p <PORT> -L 8888:localhost:8888 root@<HOST>
```

## Stop billing
Destroy or stop the instance from the Vast.ai console when done. Copy results out first.

Run order
bash

chmod +x scripts/setup_vast.sh
bash scripts/setup_vast.sh

Checklist against your acceptance criteria

    GPU + driver detected: logs/nvidia_smi.txt plus GPUs: [PhysicalDevice(...)] in the smoke test output
    Dependencies + versions recorded: logs/env_versions.txt
    Forward pass on GPU: SMOKE TEST PASSED
    Docs without secrets: docs/vast_setup.md

If tf.config.list_physical_devices("GPU") comes back empty, check that the instance’s driver supports the CUDA version bundled with TF 2.17 (CUDA 12.3). Run nvidia-smi and compare. If the driver is too old, pick a different host or lower the TF pin.


Claude is AI and can make mistakes. Please double-check responses.
Claude finished the response
