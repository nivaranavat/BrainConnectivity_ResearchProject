from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import createCorrelationMatrix, findThreshold, binarize, randomizeCorrelationMatrix, null_covariance
from source.utils.plot_utils import plot_timeseries, plot_correlation_matrix, plot_flexibility_results
from source.utils.phase_utils import *
import numpy as np
import matplotlib.pyplot as plt
import bct
import random

# Only keep workflow/analysis-specific code here. Use utils for general functions.

def main(folder_path, start=2, end=152, plot=True):
    """
    Main workflow for brain connectivity analysis using utils functions.
    """
    # Load data
    x, data = utils.readMRIFile(folder_path, size=200)
    if plot:
        utils.plot_timeseries(data, x, title="ROI Time Series", show=plot)
    # Correlation
    corr_matrix = utils.createCorrelationMatrix(data, method="pearson")
    if plot:
        utils.plot_correlation_matrix(corr_matrix, title="Correlation Matrix", show=plot)
    # Binarize
    threshold = utils.findThreshold(corr_matrix, density=0.15)
    binary_matrix = utils.binarize(corr_matrix, threshold)
    if plot:
        utils.plot_correlation_matrix(binary_matrix, title="Binarized Correlation Matrix", show=plot)
    # Network
    network = utils.createNetwork(binary_matrix)
    # Randomize
    null2 = utils.randomizeCorrelationMatrix(corr_matrix)
    null3, _ = utils.randmio_und_signed(binary_matrix, 2)
    print("Done.")
    return {
        'data': data,
        'corr_matrix': corr_matrix,
        'binary_matrix': binary_matrix,
        'network': network,
        'null2': null2,
        'null3': null3,
    }

if __name__ == "__main__":
    main("/path/to/data", start=2, end=152, plot=True)

