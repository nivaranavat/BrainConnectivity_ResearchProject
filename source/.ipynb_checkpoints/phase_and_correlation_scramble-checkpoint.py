from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import createCorrelationMatrix, findThreshold, binarize, randomizeCorrelationMatrix, null_covariance
from source.utils.plot_utils import plot_timeseries, plot_correlation_matrix, plot_flexibility_results
from source.utils.phase_utils import *
import networkx as nx
import numpy as np
import math
import random
import pandas as pd
import bct
import statistics
import matplotlib.colors
import matplotlib.pyplot as plt
    
#the goal is to create a randomized "null" of the given brain network
#have the different functions to do the different aspects of our null creation

def createCorrelationMatrix(roi_timeseries, method):
    """
    create the correlation matrix based on what the data is given and what method is used
    param: data is a 2D array of values
    param: method is the method used to find the correlation

    return: a correlation matrix
    """
    df = pd.DataFrame(roi_timeseries)
    df_transposed = df.T
    corr_matrix = df_transposed.corr(method = method)
    corr_matrix = np.array(corr_matrix)

    return corr_matrix

def calculate_sdv(vector):
    """
    param: vector is a Nx1 vector that represents a ROI
    return the standard deviation of that vector
    """
    return np.std(vector)

def getSDVofROITimeseries(roi_timeseries):

    sd = []
    for i in range(len(roi_timeseries)):
        sd.append(calculate_sdv(roi_timeseries[i]))
    return sd

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

def calculate_clustering_coef(binaryMatrix):
    """
    find the clustering coefficient of the given binarized matrix
    """

    cc_vector = bct.clustering_coef_bu(binaryMatrix) #will return a Nx1 Clustering Coefficient Vector
    ccoef = statistics.harmonic_mean(cc_vector)  #need to take the harmonic mean

    return ccoef

def calculate_path_length(binaryMatrix):
    """
    find the path length of the given binarized matrix
    """

    dist = bct.distance_bin(binaryMatrix)
    charPath,_,_,_,_ = bct.charpath(dist)

    return charPath

def calculate_small_worldness(observed, randomized):
    """
    find the small worldness ratio between the observed graph/network and the randomly rewired network
    using the ratio clustering coefficient/path length

    """

    #calculate the clustering coefficient
    observed_ccoef = calculate_clustering_coef(observed)
    randomized_ccoef = calculate_clustering_coef(randomized)
    ccoef_ratio = observed_ccoef/randomized_ccoef

    #calculate the characteristic path length
    observed_path_length = calculate_path_length(observed)
    randomized_path_length = calculate_path_length(randomized)
    path_length_ratio = observed_path_length/randomized_path_length


    #calculate the small worldness ratio between observed and random network
    small_worldness = ccoef_ratio/path_length_ratio

    return small_worldness,ccoef_ratio,path_length_ratio   

# Domain-specific phase scrambling functions (not in utils, so keep here)
def phase_scramble_1(data):
    """
    Phase scrambling: add random phase to each frequency component.
    """
    fs = np.fft.fft(data)
    pow_fs = np.abs(fs) ** 2
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    for i in range(len(phase_fsr)):
        add = np.random.uniform(0, 2 * math.pi)
        phase_fsr[i] += add
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    return tsr

def phase_scramble_2(nparray):
    """
    Phase scrambling: split phases, shuffle, and mirror.
    """
    series = pd.Series(nparray)
    fourier = np.fft.fft(series)
    pow_fs = np.abs(fourier) ** 2.
    phase_fs = np.angle(fourier)
    phase_fsr = phase_fs.copy()
    if len(nparray) % 2 == 0:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)//2]
    else:
        phase_fsr_lh = phase_fsr[1:len(phase_fsr)//2 + 1]
    np.random.shuffle(phase_fsr_lh)
    if len(nparray) % 2 == 0:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh,
                                    np.array((phase_fsr[len(phase_fsr)//2],)),
                                    phase_fsr_rh))
    else:
        phase_fsr_rh = -phase_fsr_lh[::-1]
        phase_fsr = np.concatenate((np.array((phase_fsr[0],)), phase_fsr_lh, phase_fsr_rh))
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    return tsr

    
    
def compute_correlation_matrices(time_series1, time_series2, method="pearson"):
    """
    Computes correlation matrices for two sets of time series and returns their normalized forms.
    
    Args:
        time_series1 (list): Original brain ROI time series.
        time_series2 (list): Phase-scrambled/null time series.
        method (str): Correlation method. Default is 'pearson'.

    Returns:
        tuple: (brain_corr_matrix, null_corr_matrix, diff_matrix)
    """
    brain_corr = createCorrelationMatrix(time_series1, method) - np.identity(len(time_series1))
    null_corr = createCorrelationMatrix(time_series2, method) - np.identity(len(time_series2))
    diff = brain_corr - null_corr
    return brain_corr, null_corr, diff


def plot_correlation_matrices(brain_corr, null_corr, diff_corr, save_path="Correlation_Matrix_Plots.png"):
    """
    Plots the correlation matrices: brain, null, and their difference.
    
    Args:
        brain_corr (np.ndarray): Brain correlation matrix.
        null_corr (np.ndarray): Null/scrambled correlation matrix.
        diff_corr (np.ndarray): Difference matrix.
        save_path (str): Filename to save the plot.
    """
    cmap = "hsv"
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    ax[0].matshow(brain_corr, cmap=cmap, norm=norm)
    ax[0].set_title("Brain Correlation Matrix")
    ax[0].set_ylabel("Correlation Strength")

    ax[1].matshow(null_corr, cmap=cmap, norm=norm)
    ax[1].set_title("Phase Scrambled Null Correlation Matrix")
    ax[1].set_ylabel("Correlation Strength")

    ax[2].matshow(diff_corr, cmap=cmap, norm=norm)
    ax[2].set_title("Difference between Brain and Scrambled Correlation Matrix")
    ax[2].set_ylabel("Correlation Strength")

    plt.savefig(save_path)
    plt.show()
    

def plot_small_worldness_comparison(roi_path, 
                                    method_1='phase', 
                                    method_2='correlation_randomization',
                                    threshold_range=np.arange(0.15, 0.55, 0.05),
                                    show_plot=True):
    """
    Computes and plots small worldness across a range of thresholds 
    comparing a real brain network to two null models.

    Parameters:
    - roi_path (str): Path to MRI time series data.
    - method_1 (str): First null model type ('phase').
    - method_2 (str): Second null model type ('correlation_randomization').
    - threshold_range (np.array): Array of threshold values to iterate over.
    - show_plot (bool): Whether to display the plot.

    Returns:
    - small_worldness_dict (dict): Small worldness values for both null models.
    """
    # Load brain time series and compute correlation matrix
    x, roi_timeseries = readMRIFile(roi_path, 200)
    brain_corr = createCorrelationMatrix(roi_timeseries, "pearson")

    # First null model: Phase scrambling
    if method_1 == "phase":
        null1_timeseries = [phaseScramble1(ts) for ts in roi_timeseries]
        null1_corr = createCorrelationMatrix(null1_timeseries, "pearson")
    else:
        raise ValueError("Unsupported method_1: Only 'phase' is currently implemented.")

    # Second null model: Correlation matrix randomization
    if method_2 == "correlation_randomization":
        sd = getSDVofROITimeseries(roi_timeseries)
        null2_corr = null_covariance(brain_corr, sd)
    else:
        raise ValueError("Unsupported method_2: Only 'correlation_randomization' is implemented.")

    # Helper function for thresholding and computing small-worldness
    def compute_sws_for_null(null_corr, label):
        sws_list = []
        for perc in threshold_range:
            brain_thresh = findThreshold(brain_corr, perc)
            null_thresh = findThreshold(null_corr, perc)

            brain_bin = binarize(brain_corr, brain_thresh)
            null_bin = binarize(null_corr, null_thresh)

            sws, _, _ = calculate_small_worldness(brain_bin, null_bin)
            sws_list.append(sws)

        return sws_list

    # Compute small-worldness values
    sws_null1 = compute_sws_for_null(null1_corr, "Null 1")
    sws_null2 = compute_sws_for_null(null2_corr, "Null 2")

    # Plot
    if show_plot:
        plt.plot(threshold_range, sws_null1, marker="o", label="Phase Scrambled Null")
        plt.plot(threshold_range, sws_null2, marker="o", label="Correlation Randomized Null")
        plt.xlabel("Threshold Percentage")
        plt.ylabel("Small Worldness")
        plt.title("Small Worldness vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "thresholds": threshold_range,
        "null_1": sws_null1,
        "null_2": sws_null2
    }

def run_small_worldness_analysis(
    roi_path,
    threshold_start=0.15,
    threshold_end=0.60,
    threshold_step=0.05,
    iterations=1,
    include_brain=True,
    null_models=None,
    null_labels=None,
    plot_title="Small Worldness Comparison",
    plot=True,
    save_path=None
):
    """
    Generalized function to compute and compare small worldness metrics for brain and up to two null models.

    Parameters:
    - roi_path: str, path to ROI data.
    - threshold_start, threshold_end, threshold_step: floats for thresholding.
    - iterations: number of iterations to average over.
    - include_brain: whether to include brain correlation matrix in analysis.
    - null_models: list of functions, each returning a correlation matrix given (brain_corr, roi_ts, sd).
    - null_labels: list of labels corresponding to each null model.
    - plot_title: title for the plot.
    - plot: whether to show the plots.
    - save_path: where to save the plot (if provided).
    
    Returns:
    - dict of results (avg clustering, path length, and small worldness per method)
    """

    threshold_values = np.arange(threshold_start, threshold_end + threshold_step, threshold_step)
    num_thresholds = len(threshold_values)

    # Load brain data
    _, roi_ts = readMRIFile(roi_path, 200)
    brain_corr = createCorrelationMatrix(roi_ts, "pearson")
    sd = getSDVofROITimeseries(roi_ts)

    # Setup methods
    methods = []
    labels = []

    if include_brain:
        methods.append(lambda: brain_corr)
        labels.append("Brain")

    if null_models:
        for null_fn in null_models:
            methods.append(lambda fn=null_fn: fn(brain_corr, roi_ts, sd))

    if null_labels:
        labels.extend(null_labels)

    # Storage arrays
    results = {
        label: {
            "sws": np.zeros((iterations, num_thresholds)),
            "ccoef": np.zeros((iterations, num_thresholds)),
            "path": np.zeros((iterations, num_thresholds))
        } for label in labels
    }

    for it in range(iterations):
        matrices = [method() for method in methods]

        for t_idx, perc in enumerate(threshold_values):
            brain_thresh = findThreshold(brain_corr, perc)
            brain_bin = binarize(brain_corr, brain_thresh)

            for m_idx, matrix in enumerate(matrices):
                null_thresh = findThreshold(matrix, perc)
                null_bin = binarize(matrix, null_thresh)
                sws, ccoef, path = calculate_small_worldness(brain_bin, null_bin)
                results[labels[m_idx]]["sws"][it, t_idx] = sws
                results[labels[m_idx]]["ccoef"][it, t_idx] = ccoef
                results[labels[m_idx]]["path"][it, t_idx] = path

    # Averaging
    avg_results = {}
    for label in labels:
        avg_results[label] = {
            "sws": np.mean(results[label]["sws"], axis=0),
            "ccoef": np.mean(results[label]["ccoef"], axis=0),
            "path": np.mean(results[label]["path"], axis=0)
        }

    # Plotting
    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        for label in labels:
            ax[0].plot(threshold_values, avg_results[label]["ccoef"], marker='o', label=label)
            ax[1].plot(threshold_values, avg_results[label]["path"], marker='o', label=label)
            ax[2].plot(threshold_values, avg_results[label]["sws"], marker='o', label=label)

        ax[0].set_title("Normalized Clustering Coefficient")
        ax[0].set_ylabel("Coefficient")
        ax[1].set_title("Normalized Path Length")
        ax[1].set_ylabel("Path Length")
        ax[2].set_title("Small Worldness")
        ax[2].set_ylabel("SWS")

        for a in ax:
            a.set_xlabel("Threshold Density (%)")
            a.legend()

        fig.suptitle(plot_title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        plt.show()

    return avg_results


# Example main workflow using only utils and domain-specific functions
def main(folder_path, size=200, method="pearson", density=0.15, plot=True, save_plots=False):
    x, roi_timeseries = read_mri_file(folder_path, size)
    if plot:
        plot_timeseries(roi_timeseries, x, title="Original ROI Time Series", show=plot, save_path="TimeSeries_Plots.png" if save_plots else None)
    corr_matrix = create_correlation_matrix(roi_timeseries, method)
    if plot:
        plot_correlation_matrix(corr_matrix, title="Correlation Matrix", save_path="Correlation_Matrix.png" if save_plots else None, show=plot)
    threshold = find_threshold(corr_matrix, density)
    binary_matrix = binarize_matrix(corr_matrix, threshold)
    if plot:
        plot_correlation_matrix(binary_matrix, title="Binarized Correlation Matrix", save_path="Binarized_Correlation_Matrix.png" if save_plots else None, show=plot)
    network = nx.from_numpy_array(binary_matrix)
    nx.draw(network, node_size=20, with_labels=True)
    # Example phase scrambling
    scrambled = phase_scramble_1(roi_timeseries[0])
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(x, roi_timeseries[0], label="Original")
        plt.plot(x, scrambled.real, label="Scrambled")
        plt.title("Phase Scrambling Example (First ROI)")
        plt.legend()
        if save_plots:
            plt.savefig("Phase_Scramble_Example.png")
        plt.show()
    return {
        'x': x,
        'roi_timeseries': roi_timeseries,
        'corr_matrix': corr_matrix,
        'binary_matrix': binary_matrix,
        'network': network,
        'scrambled_first_roi': scrambled
    }

    return roi_timeseries, null_timeseries
