"""
matrix_utils.py
Domain-specific utilities for correlation matrix and thresholding in brain network analysis.
"""
import numpy as np
import pandas as pd
import random
import networkx as nx

def createCorrelationMatrix(data: np.ndarray, method: str = "pearson") -> np.ndarray:
    """
    Compute correlation matrix from time series data.
    Args:
        data: np.ndarray of shape (n_rois, n_timepoints)
        method: Correlation method (default: 'pearson')
    Returns:
        np.ndarray correlation matrix
    """
    df = pd.DataFrame(data)
    df_transposed = df.T
    corr_matrix = df_transposed.corr(method=method).to_numpy()
    return corr_matrix

def findThreshold(matrix: np.ndarray, density: float = 0.15) -> float:
    """
    Find the threshold value for a matrix to keep a given density of top values.
    Args:
        matrix: np.ndarray
        density: Fraction of top values to keep (0-1)
    Returns:
        threshold value
    """
    threshold = np.percentile(matrix - np.identity(matrix.shape[0]), (1 - density) * 100)
    return threshold

def binarize(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarize a matrix given a threshold.
    Args:
        matrix: np.ndarray
        threshold: float
    Returns:
        np.ndarray (binarized)
    """
    matrix_copy = matrix.copy()
    for i in range(matrix_copy.shape[0]):
        for j in range(matrix_copy.shape[1]):
            if matrix_copy[i][j] > threshold and i != j:
                matrix_copy[i][j] = 1
            else:
                matrix_copy[i][j] = 0
    return matrix_copy

def randomizeCorrelationMatrix(matrix: np.ndarray) -> np.ndarray:
    """
    Randomize a correlation matrix by swapping values.
    Args:
        matrix: np.ndarray
    Returns:
        np.ndarray (randomized)
    """
    corr_matrix = matrix.copy()
    length = len(corr_matrix)
    for r in range(length):
        for c in range(r+1, length):
            i = random.choices([i for i in range(length)])[0]
            j = random.choices([i for i in range(length)])[0]
            if i == j:
                j = random.choices([i for i in range(length)])[0]
            corr_matrix[r][c], corr_matrix[i][j] = corr_matrix[i][j], corr_matrix[r][c]
            corr_matrix[c][r], corr_matrix[j][i] = corr_matrix[j][i], corr_matrix[c][r]
    return corr_matrix

def randomizeCorrelationMatrix(matrix,sd):
    """
    randomize the correlation matrix
    any row and column and is switched, it's corresponding pair value is also switched
    (r,c) and (c,r) are both switched
    """
    
#     null_correlation_matrix = null_covariance(matrix,sd)
#     #null_correlation_matrix = bct.null_model_und_sign(matrix)
#     return null_correlation_matrix
    

    
    corr_matrix = matrix.copy()
    length = len(corr_matrix)
    for r in range(length):
        for c in range(r+1, length):
            i = random.choices([i for i in range(length)])[0]
            j = random.choices([i for i in range(length)])[0]
            if i==j:
                j = random.choices([i for i in range(length)])[0]
            corr_matrix[r][c] , corr_matrix[i][j] = corr_matrix[i][j], corr_matrix[r][c]
            #symmetry 
            corr_matrix[c][r] , corr_matrix[j][i] = corr_matrix[j][i], corr_matrix[c][r]
    
    return corr_matrix

def createNetwork(matrix):
    """
    Create a networkx graph from a binarized matrix.
    """
    network = nx.from_numpy_matrix(matrix)
    return network 

def null_covariance(W,sd):
    
    n = len(W)
    sdd = np.diag(sd)
    w = np.dot(sdd, np.dot(W, sdd))
    e = np.mean(np.triu(w, 1))
    v = np.var(np.triu(w, 1))
    ed = np.mean(np.diag(w))
    
    m = max(2, int(np.floor((e ** 2 - ed ** 2) / v)))

    mu = np.sqrt(e / m)
    sigma = np.sqrt(-mu ** 2 + np.sqrt(mu ** 4 + v / m))

    from scipy import stats
    x = stats.norm.rvs(loc=mu, scale=sigma, size=(n, m))
    c = np.dot(x, x.T)
    a = np.diag(1 / np.diag(c))
    return np.dot(a, np.dot(c, a))