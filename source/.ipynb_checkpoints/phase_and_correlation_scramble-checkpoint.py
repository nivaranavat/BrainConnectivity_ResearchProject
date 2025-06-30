import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
import numpy.fft
import random
import bct
import math
import statistics
    
#the goal is to create a randomized "null" of the given brain network
#have the different functions to do the different aspects of our null creation
    
def readMRIFile(filename, size):

    #reads the file and creates a domain and range that will be plotted
    #returns the data, and the x values to not have to copy them over and over again


    #x = [i for i in range(0,size,2)]

    # create the x 
    x = np.arange(0,size,1)

    roi_timeseries = []
    for num in range(2,152):
        #open file
        file = filename + str(num) + '.txt'
        with open(file) as f:
            lines = f.readlines()
            y = np.loadtxt(file)
            roi_timeseries.append(y)

    return x, roi_timeseries



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


def plot_mri_and_scrambled_timeseries(folder_path, time_end, save_path="TimeSeries_Plots.png", show_plot=True, save_plot=True):
    """
    Reads MRI data from a folder, applies phase scrambling to each time series,
    and plots the original, scrambled, and their difference.

    Parameters:
    - folder_path (str): Path to the MRI files (e.g. '.../SAL_01/roi').
    - time_end (int): Upper limit of the time domain (e.g. 200).
    - save_path (str): Filename to save the figure (default: 'TimeSeries_Plots.png').
    - show_plot (bool): Whether to show the plot using plt.show().
    - save_plot (bool): Whether to save the figure.
    """
    x, roi_timeseries = readMRIFile(folder_path, time_end)
    
    # Apply phase scrambling
    null_timeseries = []
    for i in roi_timeseries:
        scrambled = phaseScramble1(i)
        null_timeseries.append(scrambled)
    
    # Create the plot
    fig, ax = plt.subplots(nrows=1 , ncols=3, figsize=(25, 5))

    # Plot original time series
    for i in roi_timeseries:
        ax[0].plot(x, i)
    ax[0].set_title("Brain Time Series")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("MRI Value")

    # Plot scrambled time series (real part only)
    for scrambled in null_timeseries:
        ax[1].plot(x, scrambled.real)
    ax[1].set_title("Phase Scrambled Null Time Series")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("MRI Value")

    # Plot difference
    for original, scrambled in zip(roi_timeseries, null_timeseries):
        ax[2].plot(x, original - scrambled.real)
    ax[2].set_title("Difference Between Brain and Scrambled Time Series")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("MRI Value")

    # Save and/or show
    if save_plot:
        plt.savefig(save_path)
    if show_plot:
        plt.show()

    return roi_timeseries, null_timeseries
    

def cell_3():
    #find the correlation matrix and plot for the observed brain network and the phase scrambled null created
    
    import matplotlib.colors
    
    cmap = "hsv"
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    
    
    #plot out the correlation matrices
    fig, ax = plt.subplots(nrows=1 , ncols=3, figsize=(25, 5))
    
    #create the correlation matrix for brain and null
    brain_correlationMatrix = createCorrelationMatrix(roi_timeseries,"pearson")
    brain_correlationMatrix = brain_correlationMatrix - np.identity(150)
    null1_correlationMatrix = createCorrelationMatrix(null_timeseries,"pearson")
    null1_correlationMatrix = null1_correlationMatrix - np.identity(150)
    
    #plot the brain's correlation matrix
    ax[0].matshow((brain_correlationMatrix-np.identity(150)),cmap = cmap, norm = norm)
    ax[0].set_title("Brain Correlation Matrix")
    ax[0].set_ylabel("Correlation Strength")
    
    #plot the null's correlation matrix
    ax[1].matshow((null1_correlationMatrix-np.identity(150)),cmap = cmap, norm = norm)
    ax[1].set_title("Phase Scrambled Null Correlation Matrix")
    ax[1].set_ylabel("Correlation Strength")
                  
    #plot the difference between brain and null correlation matrix             
    ax[2].matshow((brain_correlationMatrix-np.identity(150))-(null1_correlationMatrix-np.identity(150)),cmap = cmap, norm = norm)
    ax[2].set_title("Difference between Brain and Scrambled Correlation Matrix")
    ax[2].set_ylabel("Correlation Strength")
    
    
    #save the plots and display  
    plt.savefig("Correlation Matrix Plots.png")
    plt.show()

def cell_4():
    #binarize the networks using 15% density
    
    cmap = "binary_r"
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    
    
    #plot out the binarized matrices based on the correlation matrix and the threshold
    fig, ax = plt.subplots(nrows=1 , ncols=3, figsize=(25, 5))
    
    #binarize both brain and null correlation matrix
    brain_threshold = findThreshold(brain_correlationMatrix,0.15)
    null1_threshold = findThreshold(null1_correlationMatrix,0.15)
    brain_binaryMatrix= binarize(brain_correlationMatrix,brain_threshold)
    null1_binaryMatrix = binarize(null1_correlationMatrix,null1_threshold)
    
    print("Brain threshold: ", brain_threshold)
    print("Null1 threshold: ", null1_threshold)
    
    #plot the brain's binarized matrix
    ax[0].matshow((brain_binaryMatrix),cmap = cmap, norm = norm)
    ax[0].set_title("Brain Binarized Matrix")
    ax[0].set_ylabel("Correlation Strength")
    
    #plot the null's graph
    ax[1].matshow((null1_binaryMatrix),cmap = cmap, norm = norm)
    ax[1].set_title("Phase Scrambled Null Binarized Matrix")
    ax[1].set_ylabel("Correlation Strength")
                  
               
    #plot out the difference
    ax[2].matshow((brain_binaryMatrix)-(null1_binaryMatrix),cmap = cmap, norm = norm)
    ax[2].set_title("Difference between Brain and Scrambled Binarized Matrix")
    ax[2].set_ylabel("Correlation Strength")
            
        
    #save the plot and display it
    plt.savefig("Binarized Matrix Plots.png")
    plt.show()

def cell_5():
    
    #next step is to check the ratios of coefficient and path length between our created null and the observed brain
    
    #calculate our clustering coefficient, path length, and small worldness
    #values should be >1 or ~1
    
    small_worldness, normalized_ccoef, normalized_path_length = calculate_small_worldness(brain_binaryMatrix, null1_binaryMatrix)
    
    print()
    print("Ratios:")
    print("Brain/Null1 path length: ", normalized_path_length)
    print("Brain/Null1 clustering coefficient: " , normalized_ccoef)
    print("Brain small worldness with null1: ", small_worldness)

def cell_6():
    #try the range of 10% to 50% threshold density for binarizing the correlation matrix and plotting it
    
    
    #first collect the data for the brain
    x ,roi_timeseries = readMRIFile('/Users/niva.ranavat/Desktop/Data/SAL_01/roi',200)
    brain_correlationMatrix = createCorrelationMatrix(roi_timeseries,"pearson")
    
    #scramble the phases
    null_timeseries = []
    #doing the fourier transform on each 
    for i in range(len(roi_timeseries)):
        series = phaseScramble1(roi_timeseries[i])
        null_timeseries.append(series)
    
    #correlation matrix
    null1_correlationMatrix = createCorrelationMatrix(null_timeseries,"pearson")
    
    small_worldness_values_null1 = []
    percentage = 0.15
    
    #the loop for the range
    while percentage <= 0.50:
        
        
        # get the threshold
        brain_threshold = findThreshold(brain_correlationMatrix,percentage)
        null1_threshold = findThreshold(null1_correlationMatrix,percentage)
        
        #binarize
        brain_binaryMatrix = binarize(brain_correlationMatrix, brain_threshold)
        null1_binaryMatrix= binarize(null1_correlationMatrix,null1_threshold)
        
        
        #calculate small worldness
        small_worldness, normalized_ccoef, normalized_path_length = calculate_small_worldness(brain_binaryMatrix, null1_binaryMatrix)
        small_worldness_values_null1.append(small_worldness)
                                                                                              
        percentage += 0.05
    
        
    
    plt.plot(np.arange(0.15,0.55,0.05),small_worldness_values_null1,marker ="o")
    plt.xlabel("threshold percentage")
    plt.ylabel("small worldness")
    plt.title("Null 1 Phase scrambling Small Worldness")
    plt.show()
    
    
        
        
        

def cell_7():
    #getting the small worldness ratio for phase scrambled null and correlation matrix null and plotting it
    
    
    small_worldness_values_null2 = []
    percentage = 0.15
    
    #randomize the correlation matrix
    sd = getSDVofROITimeseries(roi_timeseries)
    null2_correlationMatrix = null_covariance(brain_correlationMatrix,sd)
    
    while percentage <= 0.50:
        
        # get the threshold
        brain_threshold = findThreshold(brain_correlationMatrix,percentage)
        null2_threshold = findThreshold(null2_correlationMatrix,percentage)
        
        #binarize
        brain_binaryMatrix = binarize(brain_correlationMatrix,brain_threshold)
        null2_binaryMatrix = binarize(null2_correlationMatrix,null2_threshold)
        
        print("With a percentage of: ", percentage, " Brain threshold: ", brain_threshold)
        print("With a percentage of: ",percentage , " Null2 threshold: ", null2_threshold)
        
        #calculate small worldness
        small_worldness, normalized_ccoef, normalized_path_length = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix)
        small_worldness_values_null2.append(small_worldness)
        
        percentage += 0.05
    
    print(small_worldness_values_null2)
    plt.plot(np.arange(0.15,0.55,0.05),small_worldness_values_null1,marker ="o",label = "Null 1")
    plt.plot(np.arange(0.15,0.55,0.05),small_worldness_values_null2,marker ="o",label = "Null 2")
    plt.xlabel("threshold percentage")
    plt.ylabel("small worldness")
    plt.legend()
    plt.title("Comparing Null 1 and Null 2")
    plt.show()
        

def cell_8():
    #average it out for some iterations to normalize the randomness
    #while also having a threshold density from 5% to 50%
    
    iterations = 25
    
    #first collect the data for the brain
    x ,roi_timeseries = readMRIFile('/Users/niva.ranavat/Desktop/Data/SAL_01/roi',200)
    brain_correlationMatrix = createCorrelationMatrix(roi_timeseries,"pearson")
    
    
    #range for the threshold percentages
    start = 0.15
    end = 0.60
    delta = 0.05
    threshold_range = int((end-start)//delta)+2
    
    #store the small worldness values
    small_worldness_values_null1 = np.zeros((iterations,threshold_range))
    small_worldness_values_null2 = np.zeros((iterations,threshold_range))
    
    #store clustering coefficient
    clustering_coef_values_null1 = np.zeros((iterations,threshold_range))
    clustering_coef_values_null2 = np.zeros((iterations,threshold_range))
    
    #store path length values
    path_length_values_null1 = np.zeros((iterations,threshold_range))
    path_length_values_null2 = np.zeros((iterations,threshold_range))
    
    for iter in range(iterations):
        
        #scramble the phases for null1
        null1_timeseries = []
        #doing the fourier transform on each 
        for i in range(len(roi_timeseries)):
            series = phaseScramble1(roi_timeseries[i])
            null1_timeseries.append(series)
    
        #correlation matrix
        null1_correlationMatrix = createCorrelationMatrix(null1_timeseries,"pearson")
    
        #randomize the correlation matrix for null 2
        sd = getSDVofROITimeseries(roi_timeseries)
        null2_correlationMatrix = randomizeCorrelationMatrix(brain_correlationMatrix,sd)
        
        percentage = start
        p_iter = 0
    
        #the loop for the range of threshold
        while percentage <= end:
    
            
             # get the threshold
            brain_threshold = findThreshold(brain_correlationMatrix,percentage)
            null1_threshold = findThreshold(null1_correlationMatrix,percentage)
            null2_threshold = findThreshold(null2_correlationMatrix,percentage)
            
            #binarize
            brain_binaryMatrix = binarize(brain_correlationMatrix,brain_threshold)
            null1_binaryMatrix = binarize(null1_correlationMatrix,null1_threshold)
            null2_binaryMatrix = binarize(null2_correlationMatrix,null2_threshold)
    
            
            
            #calculate clustering coefficient
            small_worldness_null1, normalized_ccoef_null1, normalized_path_length_null1 = calculate_small_worldness(brain_binaryMatrix, null1_binaryMatrix)
            small_worldness_null2, normalized_ccoef_null2, normalized_path_length_null2 = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix)
            
            clustering_coef_values_null1[iter][p_iter] = normalized_ccoef_null1
            clustering_coef_values_null2[iter][p_iter] = normalized_ccoef_null2
        
            
            path_length_values_null1[iter][p_iter] = normalized_path_length_null1
            path_length_values_null2[iter][p_iter] = normalized_path_length_null2
            
            small_worldness_values_null1[iter][p_iter] = small_worldness_null1
            small_worldness_values_null2[iter][p_iter] = small_worldness_null2
    
            
            percentage += delta
            p_iter += 1
    
        
    #store the small worldness values
    small_worldness_values_null1_avg = np.mean(small_worldness_values_null1,axis=0)
    small_worldness_values_null2_avg = np.mean(small_worldness_values_null2,axis=0)
    
    #store clustering coefficient
    clustering_coef_values_null1_avg = np.mean(clustering_coef_values_null1,axis=0)
    clustering_coef_values_null2_avg = np.mean(clustering_coef_values_null2,axis=0)
    
    #store path length values
    path_length_values_null1_avg = np.mean(path_length_values_null1,axis=0)
    path_length_values_null2_avg = np.mean(path_length_values_null2,axis=0)
    
    
    
    #plot the averages that was found so far
    fig,ax = plt.subplots(1,3,figsize = (25,5))
    
    ax[0].plot(np.arange(start,end+delta,delta),clustering_coef_values_null1_avg,marker='o',label = "Time Series")
    ax[0].plot(np.arange(start,end+delta,delta),clustering_coef_values_null2_avg,marker='o',label = "Correlation")
    ax[0].set_title("Normalized Clustering Coefficient")
    ax[0].set_ylabel("Normalized Clustering Coefficient")
    ax[0].set_xlabel("Threshold Density Percentage, %")
    ax[0].legend()
    
    ax[1].plot(np.arange(start,end+delta,delta),path_length_values_null1_avg,marker='o',label = "Time Series")
    ax[1].plot(np.arange(start,end+delta,delta),path_length_values_null2_avg,marker='o',label = "Correlation")
    ax[1].set_title("Normalized Path Length")
    ax[1].set_ylabel("Normalized Path Length")
    ax[1].set_xlabel("Threshold Density Percentage, %")
    ax[1].legend()
    
    
    ax[2].plot(np.arange(start,end+delta,delta),small_worldness_values_null1_avg,marker='o',label = "Time Series")
    ax[2].plot(np.arange(start,end+delta,delta),small_worldness_values_null2_avg,marker='o',label = "Correlation")
    ax[2].set_title("Small Worldness")
    ax[2].set_ylabel("Small Worldness")
    ax[2].set_xlabel("Threshold Density Percentage, %")
    ax[2].legend()     
    
    #save the figure
    plt.savefig("Average Small Worldness Plots.png")
    plt.show()
    
    print(clustering_coef_values_null2_avg)
    print(path_length_values_null2_avg)
    print(small_worldness_values_null2_avg)

def cell_9():
    #average it out for some iterations to normalize the randomness
    #while also having a threshold density from 5% to 50%
    
    iterations = 25
    
    #first collect the data for the brain
    x ,roi_timeseries = readMRIFile('/Users/niva.ranavat/Desktop/Data/SAL_01/roi',200)
    brain_correlationMatrix = createCorrelationMatrix(roi_timeseries,"pearson")
    
    
    #range for the threshold percentages
    start = 0.15
    end = 0.60
    delta = 0.05
    threshold_range = int((end-start)//delta)+2
    
    #store the small worldness values
    small_worldness_values_null2_1 = np.zeros((iterations,threshold_range))
    small_worldness_values_null2_2 = np.zeros((iterations,threshold_range))
    
    #store clustering coefficient
    clustering_coef_values_null2_1 = np.zeros((iterations,threshold_range))
    clustering_coef_values_null2_2 = np.zeros((iterations,threshold_range))
    
    #store path length values
    path_length_values_null2_1 = np.zeros((iterations,threshold_range))
    path_length_values_null2_2 = np.zeros((iterations,threshold_range))
    
    for iter in range(iterations):
        
    
        #randomize the correlation matrix for null 2
        sd = getSDVofROITimeseries(roi_timeseries)
        null2_correlationMatrix1 = null_covariance(brain_correlationMatrix,sd)
        null2_correlationMatrix2 = randomizeCorrelationMatrix(brain_correlationMatrix,sd)
        
        percentage = start
        p_iter = 0
    
        #the loop for the range of threshold
        while percentage <= end:
    
            
             # get the threshold
            brain_threshold = findThreshold(brain_correlationMatrix,percentage)
            null2_threshold1 = findThreshold(null2_correlationMatrix1,percentage)
            null2_threshold2 = findThreshold(null2_correlationMatrix2,percentage)
            
            #binarize
            brain_binaryMatrix = binarize(brain_correlationMatrix,brain_threshold)
            null2_binaryMatrix1 = binarize(null2_correlationMatrix1,null2_threshold1)
            null2_binaryMatrix2 = binarize(null2_correlationMatrix2,null2_threshold2)
    
            
            
            #calculate clustering coefficient
            small_worldness_null2_1, normalized_ccoef_null2_1, normalized_path_length_null2_1 = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix1)
            small_worldness_null2_2, normalized_ccoef_null2_2, normalized_path_length_null2_2 = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix2)
            
            clustering_coef_values_null2_1[iter][p_iter] = normalized_ccoef_null2_1
            clustering_coef_values_null2_2[iter][p_iter] = normalized_ccoef_null2_2
        
            
            path_length_values_null2_1[iter][p_iter] = normalized_path_length_null2_1
            path_length_values_null2_2[iter][p_iter] = normalized_path_length_null2_2
            
            small_worldness_values_null2_1[iter][p_iter] = small_worldness_null2_1
            small_worldness_values_null2_2[iter][p_iter] = small_worldness_null2_2
    
            
            percentage += delta
            p_iter += 1
    
        
    #store the small worldness values
    small_worldness_values_null2_1_avg = np.mean(small_worldness_values_null2_1,axis=0)
    small_worldness_values_null2_2_avg = np.mean(small_worldness_values_null2_2,axis=0)
    
    #store clustering coefficient
    clustering_coef_values_null2_1_avg = np.mean(clustering_coef_values_null2_1,axis=0)
    clustering_coef_values_null2_2_avg = np.mean(clustering_coef_values_null2_2,axis=0)
    
    #store path length values
    path_length_values_null2_1_avg = np.mean(path_length_values_null2_1,axis=0)
    path_length_values_null2_2_avg = np.mean(path_length_values_null2_2,axis=0)
    
    
    
    #plot the averages that was found so far
    fig,ax = plt.subplots(1,3,figsize = (25,5))
    
    ax[0].plot(np.arange(start,end+delta,delta),clustering_coef_values_null2_1_avg,marker='o',label = "HQS")
    ax[0].plot(np.arange(start,end+delta,delta),clustering_coef_values_null2_2_avg,marker='o',label = "Correlation")
    ax[0].set_title("Normalized Clustering Coefficient")
    ax[0].set_ylabel("Normalized Clustering Coefficient")
    ax[0].set_xlabel("Threshold Density Percentage, %")
    ax[0].legend()
    
    ax[1].plot(np.arange(start,end+delta,delta),path_length_values_null2_1_avg,marker='o',label = "HQS")
    ax[1].plot(np.arange(start,end+delta,delta),path_length_values_null2_2_avg,marker='o',label = "Correlation")
    ax[1].set_title("Normalized Path Length")
    ax[1].set_ylabel("Normalized Path Length")
    ax[1].set_xlabel("Threshold Density Percentage, %")
    ax[1].legend()
    
    
    ax[2].plot(np.arange(start,end+delta,delta),small_worldness_values_null2_1_avg,marker='o',label = "HQS")
    ax[2].plot(np.arange(start,end+delta,delta),small_worldness_values_null2_2_avg,marker='o',label = "Correlation")
    ax[2].set_title("Small Worldness")
    ax[2].set_ylabel("Small Worldness")
    ax[2].set_xlabel("Threshold Density Percentage, %")
    ax[2].legend()     
    
    #save the figure
    plt.savefig("Average Small Worldness Plots.png")
    plt.show()
    
    # print(clustering_coef_values_null2_avg)
    # print(path_length_values_null2_avg)
    # print(small_worldness_values_null2_avg)

