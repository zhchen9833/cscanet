import numpy as np


def normalize(data: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(data, dtype=np.float32)
    for c in range(data.shape[-1]):
        channel = data[:, :, c]
        if channel.max() - channel.min() != 0:
            normalized[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min())
    return normalized
