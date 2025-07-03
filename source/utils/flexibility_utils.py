"""
flexibility_utils.py
Domain-specific utilities for brain network flexibility analysis.
"""

import numpy as np
import networkx as nx
from nltools.data import Adjacency
from pathlib import Path
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

try:
    from community import community_louvain
except ImportError:
    raise ImportError("Please install python-louvain: pip install python-louvain")

from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import (
    createCorrelationMatrix,
    findThreshold,
    binarize,
    randomizeCorrelationMatrix,
    null_covariance,
    createNetwork,
    binarizeWithOutThreshold,
    calculate_sdv
)
from source.utils.phase_utils import calculate_small_worldness
from source.utils.plot_utils import box_plot

def split_roi_into_windows(roi, timepoints=60, splits=50, num_window=1):
    """
    Split the ROI time series into overlapping windows.

    Args:
        roi (np.ndarray): ROI time series (regions x timepoints).
        timepoints (int): Length of each window.
        splits (int): Number of windows to create.
        num_window (int): Step size between windows.

    Returns:
        list: List of windowed ROI arrays.
    """
    roi_split = []
    row, col = roi.shape
    start_index = 0
    for i in range(splits):
        roi_split.append(roi[:, range(start_index, start_index + timepoints)])
        start_index += num_window
    return roi_split

def calculate_threshold(roi, top=15):
    """
    Calculate a threshold for the correlation matrix based on the top percentile.

    Args:
        roi (np.ndarray): ROI time series.
        top (float): Percentile for thresholding.

    Returns:
        float: Threshold value.
    """
    roi_T = roi.T
    roi_pd = pd.DataFrame(roi_T).abs()
    corr = roi_pd.corr(method="pearson")
    threshold = np.percentile(corr, 100 - top)
    return threshold

def apply_threshold(roi_split, threshold):
    """
    Apply a threshold to a list of correlation matrices.

    Args:
        roi_split (list): List of ROI arrays (windowed).
        threshold (float): Threshold value.

    Returns:
        list: List of thresholded correlation matrices.
    """
    siz = len(roi_split)
    roi_corr = []
    for i in range(siz):
        corr = np.corrcoef(roi_split[i])
        corr[np.abs(corr) < threshold] = 0.0
        roi_corr.append(corr)
    return roi_corr

def build_graph(roi_corr, threshold):
    """
    Build a list of graphs from thresholded correlation matrices.

    Args:
        roi_corr (list): List of correlation matrices.
        threshold (float): Threshold value for graph binarization.

    Returns:
        list: List of NetworkX graphs.
    """
    graph_list = []
    siz = len(roi_corr)
    for i in range(siz):
        adj_roi = Adjacency(roi_corr[i], matrix_type='similarity')
        thresholded_roi = adj_roi.threshold(upper=threshold, binarize=False)
        g = thresholded_roi.to_graph()
        graph_list.append(g)
    return graph_list

def flexibility(list_graph):
    """
    Calculate node flexibility across a list of graphs using community detection.
    Args:
        list_graph (list): List of NetworkX graphs (one per window).
    Returns:
        np.ndarray: Flexibility score for each node.
    """
    siz = len(list_graph)
    nodes = list_graph[0].number_of_nodes()
    possible_changes = siz - 1
    cl = []
    # Initial community detection
    partition = community_louvain.best_partition(list_graph[0])
    # Convert partition dict to label array
    prev_labels = np.array([partition[n] for n in range(nodes)])
    cl.append(prev_labels)
    total_changes = np.zeros(nodes)
    for i in range(1, siz):
        partition = community_louvain.best_partition(list_graph[i])
        labels = np.array([partition[n] for n in range(nodes)])
        cl.append(labels)
        diff = (labels != cl[i-1]).astype(int)
        total_changes += diff
    return total_changes / possible_changes

def flex_with_time_shifts(drug_type, path, folder_name, timepoints=60, splits=15, shifts=[1, 3, 5, 7, 10], small_worldness_calc=False):
    """
    Calculate flexibility and (optionally) small-worldness for each time shift for a given drug.

    Args:
        drug_type (str): Drug name prefix.
        path (str): Path to data directory.
        folder_name (str): Output folder for results.
        timepoints (int): Window length.
        splits (int): Number of windows.
        shifts (list): List of window step sizes.
        small_worldness_calc (bool): Whether to calculate small-worldness.

    Returns:
        total_drug (dict): Flexibility results.
        small_worldness (dict): Small-worldness results.
        all_flexibility (dict): All flexibility values.
    """
    result_dict = defaultdict(list)
    all_flexibility = defaultdict(dict)
    for dir in Path(path).iterdir():
        if dir.is_dir():
            split_roi = [None] * len(shifts)
            drug = os.path.basename(dir)
            drug_base = drug.split('_')[0] + "_" + drug.split('_')[1]
            if drug.startswith(drug_type):
                if drug_base not in result_dict:
                    result_dict[drug_base] = defaultdict(list)
                type_drug = drug.split('_')[0]
                roi = load_txt(dir)
                threshold = calculate_threshold(roi)
                for i in range(len(shifts)):
                    split_roi[i] = split_roi_into_windows(roi, timepoints=timepoints, splits=splits, num_window=shifts[i])  
                for i in range(len(shifts)):
                    corr = apply_threshold(split_roi[i], threshold)
                    graph_list = build_graph(corr, threshold)
                    flex = flexibility(graph_list)
                    all_flexibility[shifts[i]][drug] = flex
                    # Optionally calculate small-worldness
                    sw = []
                    if small_worldness_calc:
                        for split in split_roi[i]:
                            brain_correlationMatrix = createCorrelationMatrix(split, "pearson")
                            brain_threshold = findThreshold(brain_correlationMatrix, 0.15)
                            brain_binaryMatrix = binarize(brain_correlationMatrix, brain_threshold)
                            inner_sw = []
                            for j in range(5):
                                null2_correlationMatrix = randomizeCorrelationMatrix(brain_correlationMatrix)
                                null2_threshold = findThreshold(null2_correlationMatrix, 0.15)
                                null2_binaryMatrix = binarize(null2_correlationMatrix, null2_threshold)
                                small_worldness, normalized_ccoef, normalized_path_length = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix)
                                inner_sw.append(small_worldness)
                            sw.append(np.average(inner_sw))
                    result_dict[drug_base][shifts[i]].append((i, flex, sw))
    # Save all flexibility calculations for all brain regions
    if small_worldness_calc:
        filename = "results/all_flex/" + drug_type
        if not os.path.isdir(filename):
            os.makedirs(filename)
        for i in range(len(shifts)):
            l = []
            for k, v in sorted(all_flexibility[shifts[i]].items(), key=lambda x: int(x[0].split("_")[1])):
                l.append(v)
            np.savetxt(filename + "/" + str(shifts[i]) + ".txt", np.array(l).T)
    # Average over brains and store
    total_drug = defaultdict(list)
    small_worldness = defaultdict(list)
    for base, sfts in result_dict.items():
        for index, flex in sfts.items():
            flex_list = []
            sw_list = []
            for f in flex:
                flex_list.append(f[1])
                sw_list.append(f[2])
            total_drug[base].append((f[0], np.average(flex_list, axis=0)))
            small_worldness[base].append((f[0], np.average(sw_list)))
    return total_drug, small_worldness, all_flexibility

def calculate_flexibility(data_path, drugs_list, timepoints, plot_ranges, folder_name):
    """
    Calculate flexibility for each drug and timepoint, plot results, and save to disk.

    Args:
        data_path (str): Path to data directory.
        drugs_list (list): List of drug names.
        timepoints (list): List of window lengths.
        plot_ranges (dict): Dict of plotting ranges for each drug.
        folder_name (str): Output folder for results.

    Returns:
        flexibility_calculations (dict): Flexibility statistics.
        fig (matplotlib.figure.Figure): Figure with box plots.
    """
    result_dict = dict()
    for i in range(len(drugs_list)):
        result_dict[drugs_list[i]] = defaultdict(dict)
    for dir in Path(data_path).iterdir():
        if dir.is_dir():
            split_roi = [None] * len(timepoints)
            drug = os.path.basename(dir)
            drug_base = drug.split('_')[0] + "_" + drug.split('_')[1]
            if drug.startswith(drugs_list):
                type_drug = drug.split('_')[0]
                if drug_base not in result_dict[type_drug]:
                    result_dict[type_drug][drug_base] = defaultdict(list)
                roi = load_txt(dir)
                threshold = calculate_threshold(roi)
                for i in range(len(timepoints)):
                    split_roi[i] = split_roi_into_windows(roi, timepoints=timepoints[i])
                for i in range(len(timepoints)):
                    corr = apply_threshold(split_roi[i], threshold)
                    graph_list = build_graph(corr, threshold)
                    flex = flexibility(graph_list)
                    result_dict[type_drug][drug_base][timepoints[i]].append((i, flex))
    # Average over brains and store
    total_drug = dict()
    for drug, brain in result_dict.items():
        total_drug[drug] = defaultdict(list)
        for base, tps in brain.items():
            for t, flex in tps.items():
                flex_list = []
                for f in flex:
                    flex_list.append(f[1])
                total_drug[drug][base].append((f[0], np.mean(flex_list, axis=0)))
        print(drug)
    flexibility_calculations = defaultdict(tuple)
    fig = plt.figure(figsize=(30, 50))
    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, timepoints, drugs_list[i])
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC", plot_ranges[drugs_list[i]]["mean"])
        index += 1
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3", plot_ranges[drugs_list[i]]["sdv"])
        index += 1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV", plot_ranges[drugs_list[i]]["cov"])
        index += 1
        # Save results to disk
        for key, value in fmean.items():
            filename = folder_name + "/fmean/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
        for key, value in cov.items():
            filename = folder_name + "/cov/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
        for key, value in sdv.items():
            filename = folder_name + "/sdv/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
    return flexibility_calculations, fig

def calculate_flexilibity_with_timeshifts(timepoints, splits, shifts, drugs_list, path, plot_ranges):
    """
    Calculate flexibility for each time shift for each drug in the data directory.

    Args:
        timepoints (int): Window length.
        splits (int): Number of windows.
        shifts (list): List of window step sizes.
        drugs_list (list): List of drug names.
        path (str): Path to data directory.
        plot_ranges (dict): Dict of plotting ranges for each drug.

    Returns:
        flexibility_calculations (dict): Flexibility statistics.
        fig (matplotlib.figure.Figure): Figure with box plots.
        sw (dict): Small-worldness results.
    """
    total_drug = defaultdict(list)
    sw = defaultdict(list)
    for drug in drugs_list:
        drug_dict, small_worldness, _ = flex_with_time_shifts(drug, path, timepoints, splits, shifts)
        total_drug[drug] = drug_dict
        sw[drug].append(small_worldness)
        print(drug)
    flexibility_calculations = defaultdict(tuple)
    fig = plt.figure(figsize=(30, 50))
    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, shifts, drugs_list[i])
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC", plot_ranges[drugs_list[i]]["mean"])
        index += 1
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3", plot_ranges[drugs_list[i]]["sdv"])
        index += 1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV", plot_ranges[drugs_list[i]]["cov"])
        index += 1
    return flexibility_calculations, fig, sw

def calculate_flexilibity_with_timeshifts_and_save(timepoints, splits, shifts, drugs_list, path, plot_ranges, folder_name, small_worldness_calc=False):
    """
    Calculate flexibility for each time shift for each drug and save results to disk.

    Args:
        timepoints (int): Window length.
        splits (int): Number of windows.
        shifts (list): List of window step sizes.
        drugs_list (list): List of drug names.
        path (str): Path to data directory.
        plot_ranges (dict): Dict of plotting ranges for each drug.
        folder_name (str): Output folder for results.
        small_worldness_calc (bool): Whether to calculate small-worldness.

    Returns:
        flexibility_calculations (dict): Flexibility statistics.
        fig (matplotlib.figure.Figure): Figure with box plots.
        sw (dict): Small-worldness results.
    """
    total_drug = defaultdict(list)
    sw = defaultdict(list)
    for drug in drugs_list:
        drug_dict, small_worldness, _ = flex_with_time_shifts(drug, path, folder_name, timepoints, splits, shifts, small_worldness_calc)
        total_drug[drug] = drug_dict
        sw[drug].append(small_worldness)
        print(drug)
    flexibility_calculations = defaultdict(tuple)
    fig = plt.figure(figsize=(30, 50))
    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, shifts, drugs_list[i])
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC", plot_ranges[drugs_list[i]]["mean"])
        index += 1
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3", plot_ranges[drugs_list[i]]["sdv"])
        index += 1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV", plot_ranges[drugs_list[i]]["cov"])
        index += 1
        # Save results to disk
        for key, value in fmean.items():
            filename = folder_name + "/fmean/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
        for key, value in cov.items():
            filename = folder_name + "/cov/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
        for key, value in sdv.items():
            filename = folder_name + "/sdv/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename + "/" + str(key) + ".txt", value)
    return flexibility_calculations, fig, sw


def tp_dict(result_dict, tp_or_shifts, drug=None):
    result_drug = dict()
    if drug:
        result_drug = result_dict[drug]
    else:
        result_drug = result_dict
    len_lines = len(result_drug)
    plt.figure(figsize=(10,10))
    sdv_list = list()
    cov_list = list()
    fmean_list = list()
    for index, (drug_type, list_flex) in enumerate(sorted(result_drug.items())):
        sdv = [0.0] * len(tp_or_shifts)
        cov = [0.0] * len(tp_or_shifts)
        fmean = [0.0] * len(tp_or_shifts)
        for i, flex in list_flex:
            fmean[i] = np.mean(flex)
            sdv[i] = calculate_sdv(flex)
            #need to calculate the coefficient of variance 
            cov[i] = sdv[i]/np.mean(flex)
        sdv_list.append(sdv)
        cov_list.append(cov)
        fmean_list.append(fmean)
    tp_sdv_list = list()
    tp_cov_list = list()
    tp_mean_list = list()
    for i in range(len(tp_or_shifts)):
        s = list(sdv[i] for sdv in sdv_list)
        tp_sdv_list.append(s)
        c = list(cov[i] for cov in cov_list)
        tp_cov_list.append(c)
        m = list(fmean[i] for fmean in fmean_list)
        tp_mean_list.append(m)
    temp_dict1 = dict()
    temp_dict2 = dict()
    temp_dict3 = dict()
    for i in range(len(tp_or_shifts)):
        temp_dict1[tp_or_shifts[i]] = tp_sdv_list[i]
        temp_dict2[tp_or_shifts[i]] = tp_cov_list[i]
        temp_dict3[tp_or_shifts[i]] = tp_mean_list[i]
    return (temp_dict1, temp_dict2,temp_dict3)

# Add calculate_flexibility and any other domain-specific helpers as needed, using only domain-specific logic.