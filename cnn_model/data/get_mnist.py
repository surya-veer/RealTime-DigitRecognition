import numpy as np
import os

def get_mnist(flag, one_hot = True):
    if os.path.isfile(f"../data/mnist.npz"):
        with np.load(f"./data/mnist.npz") as f:
            images, labels = f[f"x_{flag}"], f[f"y_{flag}"]
    else:
        with np.load(f"./cnn_model/data/mnist.npz") as f:
            images, labels = f[f"x_{flag}"], f[f"y_{flag}"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    if one_hot == True:
        labels = np.eye(10)[labels]
    return images, labels
