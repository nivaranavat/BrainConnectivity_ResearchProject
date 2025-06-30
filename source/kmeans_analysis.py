import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
import os
from collections import defaultdict
from nltools.data import Adjacency
import utils


def generate_synthetic_clusters(n1=50, n2=50, offset=1, scale1=-2, scale2=2):
    """
    Generate synthetic 2D clusters for testing.
    Returns: X (n_samples, 2)
    """
    X = scale1 * np.random.rand(n1 + n2, 2)
    X1 = offset + scale2 * np.random.rand(n2, 2)
    X[n1:n1+n2, :] = X1
    return X


def fit_kmeans(X, n_clusters):
    """
    Fit KMeans clustering to data X.
    Returns: fitted KMeans object
    """
    kmean = KMeans(n_clusters=n_clusters)
    kmean.fit(X)
    return kmean


def apply_threshold(roi_split, threshold, kmean):
    """
    Apply threshold to correlation matrices and return list of cluster centers.
    """
    siz = len(roi_split)
    roi_corr = []
    for i in range(siz):
        corr = np.corrcoef(roi_split[i])
        corr[np.abs(corr) < threshold] = 0.0
        roi_corr.append(kmean.cluster_centers_)
    return roi_corr


def build_graph(roi_corr, threshold):
    """
    Build graphs from thresholded correlation matrices.
    """
    graph_list = []
    siz = len(roi_corr)
    for i in range(siz):
        adj_roi = Adjacency(roi_corr[i], matrix_type='similarity')
        thresholded_roi = adj_roi.threshold(upper=threshold, binarize=False)
        g = thresholded_roi.to_graph()
        graph_list.append(g)
    return graph_list


def kmeans_flexibility_analysis(data_path, drugs_list, timepoints, n_clusters=15):
    """
    Run KMeans clustering and flexibility analysis for each drug and timepoint.
    Returns: result_dict
    """
    result_dict = dict()
    data_path = Path(data_path)
    for i in drugs_list:
        result_dict[i] = defaultdict(list)
    for dir in data_path.iterdir():
        if dir.is_dir():
            split_roi = [None] * len(timepoints)
            drug = os.path.basename(dir)
            if drug.startswith(drugs_list):
                type_drug = drug.split('_')[0]
                roi = utils.load_txt(dir)
                threshold = utils.calculate_threshold(roi)
                for i in range(len(timepoints)):
                    split_roi[i] = utils.split_roi_into_windows(roi, timepoints[i])
                for i in range(len(timepoints)):
                    kmean = KMeans(n_clusters=n_clusters)
                    kmean.fit(split_roi[i])
                    corr = apply_threshold(kmean.cluster_centers_, threshold, kmean)
                    graph_list = utils.build_graph(corr, threshold)
                    flex = utils.flexibility(graph_list)
                    result_dict[type_drug][drug].append((i, flex))
    return result_dict


def main():
    # Example synthetic data clustering
    X = generate_synthetic_clusters()
    kmean = fit_kmeans(X, n_clusters=2)
    # Example real data clustering
    x, roi = utils.readMRIFile("/Users/niva.ranavat/UCI Research/Data/SAL_01", 200)
    corr = utils.createCorrelationMatrix(roi, "pearson")
    kmean_corr = fit_kmeans(corr, n_clusters=15)
    # Example full workflow
    data_path = "/Users/niva.ranavat/UCI Research/Data"
    drugs_list = ("SAL",)
    timepoints = [30, 60, 90, 120, 150]
    result_dict = kmeans_flexibility_analysis(data_path, drugs_list, timepoints, n_clusters=15)
    return {
        'synthetic_kmeans': kmean,
        'corr_kmeans': kmean_corr,
        'result_dict': result_dict
    }

if __name__ == "__main__":
    main() 