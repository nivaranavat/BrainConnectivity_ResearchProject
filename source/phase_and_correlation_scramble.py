from source.utils.io_utils import read_mri_file, load_txt
from source.utils.matrix_utils import create_correlation_matrix, find_threshold, binarize_matrix, randomize_correlation_matrix
from source.utils.plot_utils import plot_timeseries, plot_correlation_matrix, plot_flexibility_results
import networkx as nx
import numpy as np
import math
import random
import pandas as pd
import bct
import statistics

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

if __name__ == "__main__":
    main("/path/to/data", size=200, method="pearson", density=0.15, plot=True, save_plots=False)

