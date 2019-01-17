import numpy as np
TINY = 1e-6


def standardize(x: np.array) -> np.array:
    std = np.std(x)
    if std < TINY:
        std = 1
    return (x - np.mean(x)) / std


def normalize(x: np.array) -> np.array:
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn)
