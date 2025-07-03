"""
phase_utils.py
Domain-specific utilities for phase scrambling in brain network analysis.
"""
import numpy as np
import pandas as pd
import math
import random
from source.phase_and_correlation_scramble import *
from source.utils.io_utils import readMRIFile, load_txt
from source.utils.matrix_utils import *

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


def findThreshold(corr_matrix, density):
    """
    return the threshold based on the density given for the correlation matrix
    density must be <= 1 for this function to work
    
    """
    #remove the diagonal 
    #matrix = matrix - np.identity(150)
    #find the threshold by sorting all values in matrix
    #all_values = np.array([])
    #for i in range(0,len(matrix)):
        #all_values = np.append(all_values,matrix[i][i+1:])
        
    #all_values = np.sort(all_values)
    
    #find the threshold that is at top %percentage of the matrix
    #print(all_values)
    #threshold = all_values[int(((len(matrix)*len(matrix) - len(matrix))/2)*breaking_percentage)+len(matrix)]
    #print("before",threshold,int(((len(matrix)*len(matrix) - len(matrix))/2)*breaking_percentage)+len(matrix) )
    #threshold = all_values[int(len(all_values)*(1-breaking_percentage))]
    #print("after",threshold,int(len(all_values)*(1-breaking_percentage)),all_values[int(len(all_values)*(1-breaking_percentage))])
    
    threshold = np.percentile(corr_matrix-np.identity(150), (1 - density)*100)
    return threshold
    
def binarize(matrix,threshold):
    """
    takes the matrix and binarizes it based on the percentage given% threshold
    binarizes it with values of 0 and 1
    values > threshold -> 1
    values < threshold -> 0
    returns the finished matrix at the end
    """
    
    matrix_copy = matrix.copy()
    
    #binarize matrix based on threshold
    rows = matrix_copy.shape[0]
    cols = matrix_copy.shape[1]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix_copy[i][j]>threshold and i!=j:
                matrix_copy[i][j] = 1
            else:
                matrix_copy[i][j] = 0
                
    return np.array(matrix_copy)




def createNetwork(matrix):
    """
    to visualize the network create a drawing out of it
    takes the binarized matrix and makes a network graph out of it
    returning the network variable
    """
    network = nx.from_numpy_matrix(matrix)
    nx.draw(network, node_size = 20, with_labels = True)
    return network



#two different methods to phase scramble currently

def phaseScramble1(data):
    """
    phase Scrambling the the first way we thought
    take the phases of the data given, scramble them
    then put them back into the inverse fourier
    """
    
    fs = np.fft.fft(data)
    pow_fs = np.abs(fs) ** 2
    phase_fs = np.angle(fs)
    phase_fsr = phase_fs.copy()
    
    #adding a random value between 0 and 2 Pi
    for i in range(len(phase_fsr)):
        add = np.random.uniform(0,2*math.pi)
        phase_fsr[i] += add
        
    #np.random.shuffle(phase_fsr)
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    
    #sqrt of that sum of squared of real + imag
    # sqrt of real^2+ imag^2
    
    #tsr = np.abs(tsr)
    
    return tsr


def phase_scramble(series, randomize_phase=True, rotate_phase=True, seed=None):
    """
    Phase-scramble a 1D time series using DFT.
    - randomize_phase: shuffle the phases
    - rotate_phase: add random rotation to each phase
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    fourier = DFT_slow(series)
    pow_fs = np.abs(fourier) ** 2.
    phase_fs = np.angle(fourier)
    phase_fsr = phase_fs.copy()
    if randomize_phase:
        np.random.shuffle(phase_fsr)
    if rotate_phase:
        for i in range(len(phase_fsr)):
            phase_fsr[i] += random.uniform(0, 2 * np.pi)
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    tsr = np.fft.ifft(fsrp)
    return tsr, fourier, phase_fs, phase_fsr

def phaseScramble2(nparray,n):
    """
    
    different approach but similar to previous strategy
    
    take the fourier of the time series given
    but split the phases into a left and right side and shuffle them different
    
    """
    
    series = pd.Series(nparray)
    fourier = numpy.fft.fft(series)
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
    tsr = numpy.fft.ifft(fsrp)
    
    return tsr


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

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

