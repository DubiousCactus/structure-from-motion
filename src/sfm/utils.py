import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)
