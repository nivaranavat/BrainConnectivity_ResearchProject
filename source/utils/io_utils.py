import numpy as np
import pandas as pd
from pathlib import Path
import os

def read_mri_file(folder_path: str, size: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MRI time series data from a folder.
    Args:
        folder_path: Path to the folder containing 'roi<num>.txt' files
        size: Number of timepoints
    Returns:
        x: np.ndarray of timepoints
        roi_timeseries: np.ndarray of shape (n_rois, n_timepoints)
    """
    x = np.arange(0, size, 1)
    roi_timeseries = []
    for num in range(2, 152):
        file = os.path.join(folder_path, f'roi{num}.txt')
        try:
            y = np.loadtxt(file)
            roi_timeseries.append(y)
        except Exception as e:
            print(f"Failed to load {file}: {e}")
    return x, np.array(roi_timeseries)

def load_txt(directory: str) -> np.ndarray:
    """
    Reads all files in one directory and returns a 2D array (row: one brain region, column: time).
    Args:
        directory: Path to directory with ROI text files
    Returns:
        np.ndarray of shape (n_rois, n_timepoints)
    """
    files = sorted(Path(directory).glob('roi*.txt'))
    data = [np.loadtxt(str(f)) for f in files]
    return np.array(data) 