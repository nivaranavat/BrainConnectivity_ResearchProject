import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import numpy.fft
import random
import os
import bct


def load_roi_data(folder_path, start=2, end=152):
    """
    Loads time series data from ROI text files in the given folder. 
    This function is specific to the format of the data, but you can change it if your data is formatted differently

    Parameters:
    - folder_path (str): Path to the folder containing 'roi<num>.txt' files
    - start (int): Starting ROI number (inclusive)
    - end (int): Ending ROI number (exclusive)

    Returns:
    - data (list of np.ndarray): List of loaded arrays from each ROI file
    """
    data = []
    for num in range(start, end):
        file_path = os.path.join(folder_path, f'roi{num}.txt')
        try:
            y = np.loadtxt(file_path)
            data.append(y)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
    return data

def plot_roi_data(data, x):
    """
    Plots each time series in the data list against the given x-axis.

    Parameters:
    - data (list of np.ndarray): List of y-values
    - x (np.ndarray): Common x-values for all plots
    """
    for y in data:
        plt.plot(x, y)
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.title("ROI Time Series")
    plt.show()

def process_roi_files(folder_path, start=2, end=152, plot=False):
    """
    Loads and optionally plots ROI data from text files.

    Parameters:
    - folder_path (str): Path to the folder containing 'roi<num>.txt' files
    - start (int): Starting ROI number (inclusive)
    - end (int): Ending ROI number (exclusive)
    - plot (bool): If True, plot the loaded data

    Returns:
    - data (list of np.ndarray): Loaded time series data
    """
    x = np.arange(0, 200, 1)
    data = load_roi_data(folder_path, start=start, end=end)
    if plot:
        plot_roi_data(data, x)
    return data


def compute_and_plot_correlation(data, plot=False):
    """
    Computes the Pearson correlation matrix from a list of 1D NumPy arrays (time series).
    Optionally displays a histogram and matrix plot of the correlations.

    Parameters:
    - data (list of np.ndarray): Time series data (same length) for each ROI.
    - plot (bool): If True, plot histogram and matrix heatmap of correlation values.

    Returns:
    - corr_matrix (np.ndarray): The Pearson correlation matrix.
    """
    # Convert data to DataFrame: each column = one ROI
    df = pd.DataFrame(data)
    
    # Transpose so rows = ROIs, columns = time steps
    df_transposed = df.T

    # Compute Pearson correlation matrix
    corr_matrix = df_transposed.corr(method="pearson").to_numpy()

    if plot:
        # Plot histogram of correlation values
        plt.figure(figsize=(6, 4))
        plt.hist(corr_matrix)
        plt.title("Histogram of Correlation Coefficients")
        plt.xlabel("Correlation")
        plt.ylabel("Frequency")
        plt.show()

        # Visualize correlation matrix minus identity
        plt.figure(figsize=(6, 6))
        plt.matshow(corr_matrix - np.identity(len(corr_matrix)))
        plt.title("Correlation Matrix (Off-Diagonal)", y=1.15)
        plt.colorbar()
        plt.show()
        

    return corr_matrix

def get_top_percent_threshold(matrix, percent=0.15):
    """
    Returns the threshold value corresponding to the top `percent` of off-diagonal values in the matrix.
    """
    off_diag_values = []

    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            if i != j:
                off_diag_values.append(matrix[i, j])

    # Sort in descending order
    off_diag_values.sort(reverse=True)

    # Index at which to take threshold
    cutoff_index = int(len(off_diag_values) * percent)
    threshold = off_diag_values[cutoff_index]

    return threshold


def binarize_correlation_matrix(corr_matrix, top_percent=0.15, plot=False):
    """
    Binarizes a correlation matrix by setting the top X% of values to 1 and the rest to 0.
    Optionally plots the resulting binary matrix.

    Parameters:
    - corr_matrix (np.ndarray): A square symmetric correlation matrix.
    - top_percent (float): Fraction (0–1) of top values to retain as 1s.
    - plot (bool): Whether to display a heatmap of the binary matrix.

    Returns:
    - binary_matrix (np.ndarray): Binarized version of the input matrix.
    """
    
    # Step 1: Flatten matrix to collect all pairwise correlation values
    
    all_values = []
    for i in range(0,len(corr_matrix)):
        all_values += list(corr_matrix[i])
    all_values.sort(reverse = True)
    
    # Step 2: Determine threshold to retain top `top_percent` of values
    threshold = all_values[int(((len(corr_matrix)*len(corr_matrix) - len(corr_matrix))/2)*0.15)+len(corr_matrix)]
    print("Threshold" ,threshold)
    # This doesn't include the diagonal
#     threshold = get_top_percent_threshold(corr_matrix, percent=top_percent)
#     print("Threshold" ,threshold)

    # Step 3: Create a binary matrix using the threshold
    
    rows = corr_matrix.shape[0]
    cols = corr_matrix.shape[1]
                
    binary_matrix = np.zeros_like(corr_matrix)
    for i in range(rows):
        for j in range(cols):
            if i != j and corr_matrix[i, j] > threshold:
                binary_matrix[i, j] = 1

    # Step 4: Optional visualization
    if plot:
        plt.figure(figsize=(6, 6))
        plt.matshow(binary_matrix)
        plt.title("Binarized Correlation Matrix", y=1.15)
        plt.colorbar()
        plt.show()

    return binary_matrix
    
def create_network(matrix):
    network = nx.from_numpy_matrix(matrix)
    nx.draw(network, node_size = 20, with_labels = True)
    return network   


#Methods pulled from bct and fixed due to an error
def pick_four_unique_nodes_quickly(n, seed=None):
    '''
    This is equivalent to np.random.choice(n, 4, replace=False)
    Another fellow suggested np.random.random_sample(n).argpartition(4) which is
    clever but still substantially slower.
    '''
    rng = get_rng(seed)
    k = rng.randint(n**4)
    a = k % n
    b = k // n % n
    c = k // n ** 2 % n
    d = k // n ** 3 % n
    if (a != b and a != c and a != d and b != c and b != d and c != d):
        return (a, b, c, d)
    else:
        # the probability of finding a wrong configuration is extremely low
        # unless for extremely small n. if n is extremely small the
        # computational demand is not a problem.

        # In my profiling it only took 0.4 seconds to include the uniqueness
        # check in 1 million runs of this function so I think it is OK.
        return pick_four_unique_nodes_quickly(n, rng)
        
def randmio_und_signed(R, itr, seed=None):
    '''
    This function randomizes an undirected weighted network with positive
    and negative weights, while simultaneously preserving the degree
    distribution of positive and negative weights. The function does not
    preserve the strength distribution in weighted networks.
    Parameters
    ----------
    W : NxN np.ndarray
        undirected binary/weighted connection matrix
    itr : int
        rewiring parameter. Each edge is rewired approximately itr times.
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    R : NxN np.ndarray
        randomized network
    '''
    rng = get_rng(seed)
    R = R.copy()
    n = len(R)

    itr *= int(n * (n -1) / 2)

    max_attempts = int(np.round(n / 2))
    eff = 0

    for it in range(int(itr)):
        att = 0
        while att <= max_attempts:

            a, b, c, d = pick_four_unique_nodes_quickly(n, rng)

            r0_ab = R[a, b]
            r0_cd = R[c, d]
            r0_ad = R[a, d]
            r0_cb = R[c, b]

            #rewiring condition
            if (    np.sign(r0_ab) == np.sign(r0_cd) and
                    np.sign(r0_ad) == np.sign(r0_cb) and
                    np.sign(r0_ab) != np.sign(r0_ad)):

                R[a, d] = R[d, a] = r0_ab
                R[a, b] = R[b, a] = r0_ad

                R[c, b] = R[b, c] = r0_cd
                R[c, d] = R[d, c] = r0_cb

                eff += 1
                break

            att += 1

    return R, eff

def get_rng(seed=None):
    """
    By default, or if `seed` is np.random, return the global RandomState
    instance used by np.random.
    If `seed` is a RandomState instance, return it unchanged.
    Otherwise, use the passed (hashable) argument to seed a new instance
    of RandomState and return it.
    Parameters
    ----------
    seed : hashable or np.random.RandomState or np.random, optional
    Returns
    -------
    np.random.RandomState
    """
    if seed is None or seed == np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, np.random.RandomState):
        return seed
    try:
        rstate =  np.random.RandomState(seed)
    except ValueError:
        rstate = np.random.RandomState(random.Random(seed).randint(0, 2**32-1))
    return rstate

def null_model_und_sign(W, bin_swaps=2, wei_freq=.1, seeds=None):
    '''
    This function randomizes an undirected network with positive and
    negative weights, while preserving the degree and strength
    distributions. This function calls randmio_und.m
    Parameters
    ----------
    W : NxN np.ndarray
        undirected weighted connection matrix
    bin_swaps : int
        average number of swaps in each edge binary randomization. Default
        value is 5. 0 swaps implies no binary randomization.
    wei_freq : float
        frequency of weight sorting in weighted randomization. 0<=wei_freq<1.
        wei_freq == 1 implies that weights are sorted at each step.
        wei_freq == 0.1 implies that weights sorted each 10th step (faster,
            default value)
        wei_freq == 0 implies no sorting of weights (not recommended)
    seed : hashable, optional
        If None (default), use the np.random's global random state to generate random numbers.
        Otherwise, use a new np.random.RandomState instance seeded with the given value.
    Returns
    -------
    W0 : NxN np.ndarray
        randomized weighted connection matrix
    R : 4-tuple of floats
        Correlation coefficients between strength sequences of input and
        output connection matrices, rpos_in, rpos_out, rneg_in, rneg_out
    Notes
    -----
    The value of bin_swaps is ignored when binary topology is fully
        connected (e.g. when the network has no negative weights).
    Randomization may be better (and execution time will be slower) for
        higher values of bin_swaps and wei_freq. Higher values of bin_swaps
        may enable a more random binary organization, and higher values of
        wei_freq may enable a more accurate conservation of strength
        sequences.
    R are the correlation coefficients between positive and negative
        strength sequences of input and output connection matrices and are
        used to evaluate the accuracy with which strengths were preserved.
        Note that correlation coefficients may be a rough measure of
        strength-sequence accuracy and one could implement more formal tests
        (such as the Kolmogorov-Smirnov test) if desired.
    '''
    rng = get_rng(seeds)
    if not np.allclose(W,W.T):
        raise KeyError("Input must be undirected")
    W = W.copy()
    n = len(W)
    np.fill_diagonal(W, 0)  # clear diagonal
    Ap = (W > 0)  # positive adjmat
    An = (W < 0)  # negative adjmat

    if np.size(np.where(Ap.flat)) < (n * (n - 1)):
        W_r, eff = randmio_und_signed(W,bin_swaps,seed = rng)
        Ap_r = W_r > 0
        An_r = W_r < 0
    else:
        Ap_r = Ap
        An_r = An

    W0 = np.zeros((n, n))
    for s in (1, -1):
        if s == 1:
            Acur = Ap
            A_rcur = Ap_r
        else:
            Acur = An
            A_rcur = An_r

        S = np.sum(W * Acur, axis=0)  # strengths
        Wv = np.sort(W[np.where(np.triu(Acur))])  # sorted weights vector
        i, j = np.where(np.triu(A_rcur))
        Lij, = np.where(np.triu(A_rcur).flat)  # weights indices

        P = np.outer(S, S)

        if wei_freq == 0:  # get indices of Lij that sort P
            Oind = np.argsort(P.flat[Lij])  # assign corresponding sorted
            W0.flat[Lij[Oind]] = s * Wv  # weight at this index
        else:
            wsize = np.size(Wv)
            wei_period = np.round(1 / wei_freq)  # convert frequency to period
            lq = np.arange(wsize, 0, -wei_period, dtype=int)
            for m in lq:  # iteratively explore at this period
                # get indices of Lij that sort P
                Oind = np.argsort(P.flat[Lij])
                #[np.array([i for i in range(min(m.item(),wei_period))])]
                R = rng.permutation(m)[:min(m.item(),int(wei_period.item()))]
                for q, r in enumerate(R):
                    # choose random index of sorted expected weight
                    o = Oind[r]
                    W0.flat[Lij[o]] = s * Wv[r]  # assign corresponding weight

                    # readjust expected weighted probability for i[o],j[o]
                    f = 1 - Wv[r] / S[i[o]]
                    P[i[o], :] *= f
                    P[:, i[o]] *= f
                    f = 1 - Wv[r] / S[j[o]]
                    P[j[o], :] *= f
                    P[:, j[o]] *= f

                    # readjust strength of i[o]
                    S[i[o]] -= Wv[r]
                    # readjust strength of j[o]
                    S[j[o]] -= Wv[r]

                O = Oind[R]
                # remove current indices from further consideration
                Lij = np.delete(Lij, O)
                i = np.delete(i, O)
                j = np.delete(j, O)
                Wv = np.delete(Wv, R)

    W0 = W0 + W0.T

    rpos_in = np.corrcoef(np.sum(W * (W > 0), axis=0),
                          np.sum(W0 * (W0 > 0), axis=0))
    rpos_ou = np.corrcoef(np.sum(W * (W > 0), axis=1),
                          np.sum(W0 * (W0 > 0), axis=1))
    rneg_in = np.corrcoef(np.sum(-W * (W < 0), axis=0),
                          np.sum(-W0 * (W0 < 0), axis=0))
    rneg_ou = np.corrcoef(np.sum(-W * (W < 0), axis=1),
                          np.sum(-W0 * (W0 < 0), axis=1))
    return W0, (rpos_in[0, 1], rpos_ou[0, 1], rneg_in[0, 1], rneg_ou[0, 1])


def randomize_correlation_matrix(matrix):
    corr_matrix = matrix.copy()
    length = len(corr_matrix)
    for r in range(length):
        for c in range(r+1, length):
    #         i = random.randint(0,length-1)
    #         j = random.randint(0,length-1)
    #         if i==j:
    #             j = random.randint(0,length-1)
            i = random.choices([i for i in range(length)])[0]
            j = random.choices([i for i in range(length)])[0]
            if i==j:
                j = random.choices([i for i in range(length)])[0]
            corr_matrix[r][c] , corr_matrix[i][j] = corr_matrix[i][j], corr_matrix[r][c]
            #symmetry 
            corr_matrix[c][r] , corr_matrix[j][i] = corr_matrix[j][i], corr_matrix[c][r]
    return corr_matrix


def cell_9():
    series = pd.Series(data[0], index = x)
    plt.plot(series) 
    print("the original series") 
    plt.show()
    
    #fourier = DFT_slow(series)
    #print(fourier)
    fourier = numpy.fft.rfft(series,norm = "ortho")
    
    #freq = np.fft.fftfreq(x.shape[-1])
    #plt.plot(freq, fourier.real, freq, fourier.imag)
    print(len(fourier)) 
    print(fourier)
    #plt.show()
    
    #plt.plot(x, fourier.real, 'b-', x, fourier.imag, 'r--')
    plt.plot(fourier) 
    print("fourier transform") 
    plt.show()
    
    #pow_fs = fourier[1:-1:2]2 + fourier[2::2]2
    #phase_fs = np.arctan2(fourier[2::2], fourier[1:-1:2])
    pow_fs = np.abs(fourier) ** 2 
    phase_fs = np.angle(fourier) 
    
    print("phases")
    plt.plot(phase_fs)
    plt.show()
    phase_fsr = phase_fs.copy() 
    print(type(phase_fsr)) 
    print(phase_fsr) 
    print("before shuffled") 
    plt.plot(phase_fsr) 
    plt.show()
    
    print(len(phase_fsr))
    print(type(phase_fsr))
    # phase_fsr_lh = phase_fsr[1:len(phase_fsr)//2] print(len(phase_fsr_lh)) 
    #np.random.shuffle(phase_fsr_lh) 
    #phase_fsr_rh = -phase_fsr_lh[::-1] print(len(phase_fsr_rh)) 
    #phase_fsr = np.append(phase_fsr[0], np.append(phase_fsr_lh, np.append(phase_fsr[len(phase_fsr)//2], phase_fsr_rh)))
    # print(len(phase_fsr))
    # phase_fsr = phase_fsr+phase_fsr[0]
    # print(len(phase_fsr))
    # phase_fsr_lh = phase_fsr[1:len(phase_fsr)/2]
    # np.random.shuffle(phase_fsr_lh)
    # phase_fsr_rh = -phase_fsr_lh[::-1]
    # phase_fsr = np.append(phase_fsr[0],np.append(phase_fsr_lh, np.append(phase_fsr[len(phase_fsr)/2],phase_fsr_rh)))
    
    #np.random.shuffle(phase_fsr)
    # for i in range(len(phase_fsr)):
    # rotate = random.uniform(0,2*math.pi)
    # phase_fsr[i]+=rotate

def cell_10():
    print("shuffled")
    plt.plot(phase_fsr)
    plt.show()
    #     fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    #     fsrp = np.r_[fourier[0], fsrp.ravel(), fourier[-1]]
    fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
    #plt.plot(x, fsrp.real, 'b-', x, fsrp.imag, 'r--')
    plt.plot(fsrp)
    plt.show()
    tsr = numpy.fft.irfft(fsrp)
    print(tsr)
    plt.plot(x, tsr.real, 'b-', x, tsr.imag, 'r--')
    print(data[0])
    plt.plot(tsr)
    print("reverted the randomized using inverse fourier")
    plt.show()

def cell_11():
    #null1 
    import math
    for i in range(len(data)):
            plt.plot(x,data[i])
    plt.show()
    
    #fix the number of plots in time series something is getting cut
    #going from 200 to 100 problem
    #most likely a normalization problem, 
    #make it so that the units on every plot is the same and the number of plots is the same
    def DFT_slow(x):
        """Compute the discrete Fourier Transform of the 1D array x"""
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, x)
    
    
    
    def scramblingTimeSeries(nparray,n):
        #all_phases = []
        #for i in range(len(data)):
        series = pd.Series(nparray, index = x)
        fourier = DFT_slow(series)
        #fourier = numpy.fft.fft(series,norm = "ortho")
        pow_fs = np.abs(fourier) ** 2.
        phase_fs = np.angle(fourier)
        phase_fsr = phase_fs.copy()
        if(n%2==0):
            np.random.shuffle(phase_fsr)
        for i in range(len(phase_fsr)):
            rotate = random.uniform(0,2*math.pi)
            phase_fsr[i]+=rotate
        fsrp = np.sqrt(pow_fs) * (np.cos(phase_fsr) + 1j * np.sin(phase_fsr))
        tsr = numpy.fft.ifft(fsrp)
        plt.plot(x, tsr.real, 'b-', x, tsr.imag, 'r--')
        #plt.plot(tsr)
        #plt.show()
        return tsr
    
    
    
    timeSeries = []
    #doing the fourier transform on each 
    for i in range(len(data)):
            series = scramblingTimeSeries(data[i],i)
            timeSeries.append(series)
    plt.show()
    

def cell_12():
    import pywt
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    x = np.arange(0,200)
    
    newTimeSeries = []
    matrix = []
    fig,ax = plt.subplots(2)
    for i in range(2,152):
        data = np.loadtxt('/Users/nivaranavat/UCI Research/Data/SAL_01/roi'+str(i)+'.txt')
        (cA,cD) = pywt.dwt(data,'db2')
        randomized = np.random.permutation(cD)
        newSignal = pywt.idwt(cA,randomized,'db2')
        newTimeSeries.append(newSignal)
        matrix.append(data)
        ax[0].plot(data)
        ax[1].plot(newSignal)
    matrix = np.array(matrix)
    for i in range(0,200):
        col = matrix[:,i]
        matrix[:,i] = np.random.permutation(col)
    plt.plot(matrix)
    plt.show()
    # plt.plot(data, label = "original")
    # plt.legend()
    # plt.show()
    
    # print(cD)
    # plt.plot(cD,label = "transformed")
    # plt.legend()
    # plt.show()
    
    # plt.plot(randomized,label = "randomized")
    # plt.legend()
    # plt.show()
    
    # plt.plot(newSignal, label = "newsignal")
    # print(data)
    # print(newSignal)
    # plt.legend()
    # plt.show()

def cell_13():
    df = pd.DataFrame(matrix)
    df_transposed = df.T
    corr_matrix = df_transposed.corr(method = "pearson")
    corr_matrix = np.array(corr_matrix)
    plt.hist(corr_matrix)
    plt.matshow(corr_matrix-np.identity(150))
    plt.show()

def cell_14():
    df = pd.DataFrame(newTimeSeries)
    df_transposed = df.T
    null1 = df_transposed.corr(method = "pearson")
    null1 = np.array(null1)
    plt.hist(null1)
    plt.matshow(null1 - np.identity(150))
    plt.matshow((corr_matrix-np.identity(150))-(null1-np.identity(150)))
    plt.show()

def cell_15():
    #null1 creating the network
    from scipy.stats import pearsonr
    df_null1 = pd.DataFrame(timeSeries)
    df_transposed = df.T
    null1 = df_transposed.corr()
    null1 = np.array(null1)
    corr = np.correlate(data[0], data[1])
    result = numpy.correlate(data[0], data[0], mode='full')
    #result = result[result.size/2:]
    print(result)
    print(len(result))
    print(corr_matrix)
    print(null1)
    plt.matshow(null1-np.identity(150))
    plt.show()
    plt.matshow((corr_matrix-np.identity(150))-(null1-np.identity(150)))
    plt.show()
    plt.hist(corr_matrix)
    plt.show()
    plt.hist(null1)
    plt.show()
    binarize(null1)
    null1_network = createNetwork(null1)

def cell_16():
    #next step is to check the degree preservation
    #make sure that the sorted list of degrees of null versus brain network are the same
    #after that find the clustering coefficient for all networks 
    #equation is cc(brain)/cc(null)
    #do it for each null
    #do same for path lengths
    #values should be >1 or ~1
    #then to calculate the small worldness by doing clustering coefficient/pathlength
    
    brain_degree = sorted(matrix.sum(axis=1))
    null3_degree = sorted(null3.sum(axis=1))
    null2_degree = sorted(null2.sum(axis=1))
    null1_degree = sorted(null1.sum(axis=1))
    
    
    plt.hist(brain_degree)
    plt.show()
    #plt.matshow(brain_degree)
    plt.hist(null3_degree)
    plt.show()
    plt.hist(null2_degree)
    plt.show()
    plt.hist(null1_degree)
    plt.show()
    
    preserved = True
    for i in range(len(brain_degree)):
        if brain_degree[i] != null3_degree[i]:
            print(f"Error: The degree of the null3 network has NOT been preserved, degree {brain_degree[i]} is not preserved")
            preserved = False
            break 
        elif brain_degree[i] != null2_degree[i]:
            print(f"Error: The degree of the null2 network has NOT been preserved, degree {brain_degree[i]} is not preserved")
            preserved = False
            break
    
    if preserved:
        print("The degree of the networks has been preserved") 
    
    print()
    
    brain_coef = bct.clustering_coef_bu(matrix).sum()
    print("brains clustering coefficient", brain_coef)
    null3_coef = bct.clustering_coef_bu(null3).sum()
    print("null3 clustering coefficient", null3_coef)
    null2_coef = bct.clustering_coef_bu(null2).sum()
    print("null2 clustering coefficient", null2_coef)
    null1_coef = bct.clustering_coef_bu(null1).sum()
    print("null1 clustering coefficient", null1_coef)
    
    print()
    
    
    brain_charPath = bct.charpath(bct.distance_bin(matrix))[0]
    print("brains path length", brain_charPath)
    null3_charPath = bct.charpath(bct.distance_bin(null3))[0]
    print("null3 path length", null3_charPath)
    null2_charPath = bct.charpath(bct.distance_bin(null2))[0]
    print("null2 path length", null2_charPath)
    null1_charPath = bct.charpath(bct.distance_bin(null1))[0]
    print("null1 path length", null1_charPath)
    
    print()
    print("Ratios:")
    print("Brain/Null3 path length: ", brain_charPath/null3_charPath)
    print("Brain/Null2 path length: ", brain_charPath/null2_charPath)
    print("Brain/Null1 path length: ", brain_charPath/null1_charPath)
    print("Brain/Null3 clustering coefficient: ", brain_coef/null3_coef)
    print("Brain/Null2 clustering coefficient: " , brain_coef/null2_coef)
    print("Brain/Null1 clustering coefficient: " , brain_coef/null1_coef)
    print()
    print("Brain small worldness with null3: ", (brain_coef/null3_coef)/(brain_charPath/null3_charPath))
    print("Brain small worldness with null2: ", (brain_coef/null2_coef)/(brain_charPath/null2_charPath))
    print("Brain small worldness with null1: ", (brain_coef/null1_coef)/(brain_charPath/null1_charPath))
    
    

def run_connectivity(folder_path, start=2, end=152):
    
    data = process_roi_files(folder_path, start=start, end=end)
    corr_matrix = compute_and_plot_correlation(data)
    
    matrix = corr_matrix.copy()
    binary_matrix = binarize_correlation_matrix(matrix, plot = True)
    binary_matrix = np.array(binary_matrix)
    
    network = create_network(matrix)

