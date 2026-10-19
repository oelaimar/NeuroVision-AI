import sys
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.sysconfig.get_build_info().get("cuda_version"))
print("GPUs:", gpus)

if not gpus:
    sys.exit("fail: no GPU detected by TensorFlow")

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