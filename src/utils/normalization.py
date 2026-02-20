import numpy as np

def zscore_with_stats(arr: np.ndarray, mean: float, std: float, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize an array using precomputed statistics.

    Returns:
        normalized array
    """
    return (arr - mean) / (std + eps)
