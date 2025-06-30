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

try:
    from community import community_louvain
except ImportError:
    raise ImportError("Please install python-louvain: pip install python-louvain")

def split_roi_into_windows(roi, timepoints=60, splits=50, num_window=1):
    roi_split = []
    row, col = roi.shape
    start_index = 0
    for i in range(splits):
        roi_split.append(roi[:, range(start_index, start_index + timepoints)])
        start_index += num_window
    return roi_split

def build_graph(roi_corr, threshold):
    graph_list = []
    siz = len(roi_corr)
    for i in range(siz):
        adj_roi = Adjacency(roi_corr[i], matrix_type='similarity')
        thresholded_roi = adj_roi.threshold(upper=threshold, binarize=False)
        g = thresholded_roi.to_graph()
        graph_list.append(g)
    return graph_list

def flexibility(list_graph):
    siz = len(list_graph)
    nodes = list_graph[0].number_of_nodes()
    graph_array = [nx.convert_matrix.to_numpy_array(g) for g in list_graph]
    possible_changes = siz - 1
    cl = []
    initial_modularity = community_louvain(graph_array[0], gamma=1, ci=None, B='modularity', seed=None)
    cl.append(initial_modularity[0])
    total_changes = np.zeros(nodes)
    for i in range(1, siz):
        next_modularity = community_louvain(graph_array[i], ci=cl[i-1], B='modularity', seed=None)
        cl.append(next_modularity[0])
        diff = np.abs(next_modularity[0] - cl[i-1])
        diff[diff>0] = 1
        total_changes += diff
    return total_changes / possible_changes

# Add calculate_flexibility and any other domain-specific helpers as needed, using only domain-specific logic. 