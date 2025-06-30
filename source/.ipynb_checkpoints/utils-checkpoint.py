import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})


import pandas as pd
import networkx as nx
import numpy.fft
import random

import bct 
from bct.algorithms import community_louvain, ci2ls

import math
import statistics
from statistics import mean

from sklearn.metrics import pairwise_distances
import seaborn as sns
from nltools.data import Adjacency

from pathlib import Path 
import os
from collections import defaultdict

import sys #to use unicode for the greek letters

from sklearn.cluster import KMeans



drug_names = {"SAL" : "Saline" , "COC" : "Cocaine", "MDPV" : "MDPV", "RANDOM": "Random"} #mapping of the abbreivated name to full name of drug

#the goal is to create a randomized "null" of the given brain network
#have the different functions to do the different aspects of our null creation


"""
Functions to create the phase scrambled time series based on the original datsa

"""

def readMRIFile(filename, size):

    #reads the file and creates a domain and range that will be plotted
    #returns the data, and the x values to not have to copy them over and over again
    
    
    # create the x 
    x = np.arange(0,size,1)
    
    roi_timeseries = []
    for num in range(2,152):
        #open file
        file = filename + "/roi" + str(num) + '.txt'
        with open(file) as f:
            #lines = f.readlines()
            y = np.loadtxt(f)
            roi_timeseries.append(y)
            
    return x, np.array(roi_timeseries)



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
    
    threshold = np.percentile(corr_matrix, (1 - density)*100)
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

def phaseScramble2(nparray):
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



def randomizeCorrelationMatrix(matrix):
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
    small_worldness, ccoef_ratio, path_length_ratio = 0, 0, 0
    
    #calculate the clustering coefficient
    observed_ccoef = calculate_clustering_coef(observed)
    randomized_ccoef = calculate_clustering_coef(randomized)
    #print(observed_ccoef,randomized_ccoef)
    if (abs(randomized_ccoef) <= 0.0001): 
        #print("Clustering Coef is 0")
        pass
    else: ccoef_ratio = observed_ccoef/randomized_ccoef
    
    #calculate the characteristic path length
    observed_path_length = calculate_path_length(observed)
    randomized_path_length = calculate_path_length(randomized)
    #print(observed_path_length,randomized_path_length)
    if(abs(randomized_path_length) <= 0.0001):
        #print("Path Length is 0")
        pass
    else: path_length_ratio = observed_path_length/randomized_path_length
    
    
    #calculate the small worldness ratio between observed and random network
    small_worldness = ccoef_ratio/path_length_ratio
    
    return small_worldness, ccoef_ratio, path_length_ratio

def phaseScrambled(drug_list, path, filesize = 200, iterations = 5, threshold = 0.15):
    """
    phase scramble and calculate the small worldness at the same time of the given drug
    
    
    """
    sw = defaultdict(list)
    
    for dir in Path(path).iterdir():
        if dir.is_dir():
            drug = os.path.basename(dir)
            if drug.startswith(drug_list):
                type_drug = drug.split('_')[0]
                print(drug)
                roi = load_txt(dir)
                    
            
                #scramble the phases of the time series of the drug
                for i in range(len(roi)):
                    series = []
                    for j in range(iterations):
                        s = phaseScramble1(roi[i])
                        s2 = phaseScramble2(roi[i]) #original static randomness
                        series.append(s)
                        #save our data to our folder
                        directory = path + "/SCR" + type_drug + "_" + drug.split("_")[1] + "_" + str(j+1) #ex: SAL_01_1, SAL_01_2 ... 
                        directory2 = path + "/OrgSCR" + type_drug + "_" + drug.split("_")[1] + "_" + str(j+1)

                        if not os.path.isdir(directory):
                            os.makedirs(directory)
                            
                        if not os.path.isdir(directory2):
                            os.makedirs(directory2)
                        
                        np.savetxt(directory + "/roi" + str(i+2) + ".txt", np.absolute(s))
                        np.savetxt(directory2 + "/roi" + str(i+2) + ".txt", np.absolute(s2))

                    


def phaseScramble_and_smallWorldness(drug_list, path, filesize = 200, iterations = 5, threshold = 0.15):
    """
    phase scramble and calculate the small worldness at the same time of the given drug
    
    
    """
    sw = defaultdict(list)
    
    for dir in Path(path).iterdir():
        if dir.is_dir():
            drug = os.path.basename(dir)
            if drug.startswith(drug_list):
                type_drug = drug.split('_')[0]
                print(drug)
                roi = load_txt(dir)
                    
                scrambled_timeseries = []
            
                #scramble the phases of the time series of the drug
                for i in range(len(roi)):
                    series = []
                    for j in range(iterations):
                        s = phaseScramble1(roi[i])
                        series.append(s)
                        #save our data to our folder
                        directory = path + "/SCR" + type_drug + "_" + drug.split("_")[1] + "_" + str(j+1) #ex: SAL_01_1, SAL_01_2 ... 
                        if not os.path.isdir(directory):
                            os.makedirs(directory)
                        
                        np.savetxt(directory + "/roi" + str(i+2) + ".txt", np.absolute(s))
                        
                    
                    #need to take average to calculate small worldness
                    avg_series = np.average(series, axis = 0)
                    scrambled_timeseries.append(avg_series)
                    
                    
            
                #small worldness is calculated based on the average over num iterations

                #formulate our correlation matrix
                brain_correlationMatrix = createCorrelationMatrix(roi, "pearson")
                null1_correlationMatrix = createCorrelationMatrix(scrambled_timeseries,"pearson")

                brain_threshold = findThreshold(brain_correlationMatrix,threshold)
                null1_threshold = findThreshold(null1_correlationMatrix,threshold)

                #binarize
                brain_binaryMatrix = binarize(brain_correlationMatrix, brain_threshold)
                null1_binaryMatrix = binarize(null1_correlationMatrix,null1_threshold)


                #calculate small worldness
                small_worldness, normalized_ccoef, normalized_path_length  = calculate_small_worldness(brain_binaryMatrix, null1_binaryMatrix)
                
                sw[type_drug].append(small_worldness)
                
                
    return sw      


#how should we calculate the small worldness of the random series what should it be compared to
#def getSmallWorldnessofTimeSeries(observed_timeseries, control_timeseries)
                    
                


"""
Flexibility and Modularity functions

"""

def load_txt(directory):
#     This function reads all files in one directory 
#     and returns 150 * 200 array (there are some exceptions) (row: one brain region, column: time)
    dir_name = str(directory)
    roi = list()
    if directory.is_dir():
        for i in range(2, 152):
            file = dir_name + "/" + "roi" + str(i) + ".txt"
            roi.append(np.genfromtxt(file, dtype = np.float32)) #complex because the phase scrambled values are complex
    return np.array(roi)


def split_roi_into_windows1(roi, timepoints=60, splits=50):
    roi_split = []
    row, col = roi.shape
    num_windows = int(row / splits) #number of windows needed to skip
    start_index = 0
    for i in range(splits):
        if i == splits - 1:
            roi_split.append(roi[:, range(row-timepoints, row)])
        else:
            roi_split.append(roi[:, range(start_index, start_index + timepoints)])
        start_index += num_windows
    return roi_split 


def split_roi_into_windows(roi, timepoints=60, splits=50, num_window=1):
    roi_split = []
    row, col = roi.shape
    start_index = 0
    for i in range(splits):
        roi_split.append(roi[:, range(start_index, start_index + timepoints)])
        start_index += num_window
    return roi_split 


def calculate_threshold(roi, top=15):
    #calculate threshold 
    roi_T = roi.T
    roi_pd = pd.DataFrame(roi_T).abs()
    corr = roi_pd.corr(method="pearson") #calculate corration using pearson
    threshold = np.percentile(corr, 100 - top)
    return threshold


def apply_threshold(roi_split, threshold):
    siz = len(roi_split)
    roi_corr = []
    for i in range(siz):
        corr = np.corrcoef(roi_split[i])
        corr[np.abs(corr) < threshold] = 0.0
        roi_corr.append(corr)
    return roi_corr




def build_graph(roi_corr, threshold):
    # Build graph 
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
    #convert nx to np.array 
    graph_array = list(nx.convert_matrix.to_numpy_array(list_graph[i]) for i in range(siz))
    possible_changes = siz - 1
    cl = list()
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



def calculate_sdv(arr: np.ndarray):
    return np.std(arr)


def calculate_cov(arr: np.ndarray):
    return np.cov(arr)


def calculate_var(arr: np.ndarray):
    return np.var(arr)



def calculate_flexibility(data_path, drugs_list, timepoints, plot_ranges, folder_name):
    """
    will take the given data and formulate the flexibility based on the given time_points
    have the data have the drug or any material needed in the name of each file
    
    param: data_path is a string that tells the path to the folder with all the data
    param: drugs_list is a list specifies the names of the drugs used in this experiment
    param: timepoints is a list of time window lengths we want to test on
    
    
    """
    result_dict = dict() #will be a 3d dictionary drug->drug folder name(specific brain)->timepoint -> flexibility

    for i in range(len(drugs_list)):
        result_dict[drugs_list[i]] = defaultdict(dict) #defaultdict(np.ndarray)
    
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
                    split_roi[i] = split_roi_into_windows(roi, timepoints = timepoints[i])
                for i in range(len(timepoints)):
                    corr = apply_threshold(split_roi[i], threshold)
                    graph_list = build_graph(corr, threshold)
                    flex = flexibility(graph_list)
                    result_dict[type_drug][drug_base][timepoints[i]].append((i, flex))
         
    #print(result_dict)
    total_drug = dict()
    #take the average over each brain and store it
    for drug, brain in result_dict.items():
        total_drug[drug] = defaultdict(list)
        for base,tps in brain.items():
            for t, flex in tps.items():
                flex_list = []
                for f in flex:
                    flex_list.append(f[1])
                #write it out into a text file before taking average -> 150 by 12
                total_drug[drug][base].append((f[0], np.mean(flex_list,axis = 0)))
        print(drug)
    #print(total_drug)

    flexibility_calculations = defaultdict(tuple)
    fig = plt.figure(figsize = (30,50))
    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, timepoints, drugs_list[i])
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC", plot_ranges[drugs_list[i]]["mean"])
        index+=1
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3",plot_ranges[drugs_list[i]]["sdv"])
        index+=1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV",plot_ranges[drugs_list[i]]["cov"])
        index+=1
        
        #add the data to a text file
        
       
        for key, value in fmean.items(): 
            filename  = folder_name+"/fmean/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)

        for key, value in cov.items(): 
            filename  = folder_name+"/cov/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)

        for key, value in sdv.items(): 
            filename  = folder_name+"/sdv/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)
        
    return flexibility_calculations, fig




def calculate_kmeans(data, n_clusters):
    """
    find the k-means clustering based on the given number of clusters on the given data
    """
    kmean = KMeans(n_clusters = n_clusters)
    kmean.fit(data)
    return kmean
                    
    
                    
#def flexibility_using_kmeans()
                    
    
    
#Now result_dict stores all flexibilities
#e.g. result_dict["COL"]["COL_01"][0] = COL_01 flexibility ndarray with timepoints 30
#check standard deviation
def print_sdv(result_dict, tp_or_shifts, drug=None):
    result_drug = dict()
    if drug:
        result_drug = result_dict[drug]
    else:
        result_drug = result_dict
    for drug_type, list_flex in sorted(result_drug.items()):
        print(drug_type)
        for i, flex in list_flex:
            sdv = calculate_sdv(flex)
            print("tp_or_shifts " + str(tp_or_shifts[i]) + ": " + str(sdv))
        print()
        
        
        
        
        
#create panda Dataframe 
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



#graph out the boxplot of our data
def box_plot(tp_dict,subplot_index, figure, drug_type, y_value, y_range):
    df = pd.DataFrame(data=tp_dict)
    row, col, index = subplot_index
    ax = figure.add_subplot(row, col, index)
    
    name = drug_type
    if drug_type in drug_names:
        name = drug_names[drug_type]
    elif "OrgSCR" in drug_type:
        name = "\u03BB''" + drug_names[name.split("OrgSCR")[1]]
    elif "SCR" in drug_type:
        name = "\u03BB'" + drug_names[name.split("SCR")[1]]
    

    ax.title.set_text((name + " " + y_value))
    ax.set_ylim(y_range)
    #will need to set the range here
    sns.boxplot(ax = ax, 
                data=df, 
                width=0.5, 
                color='lightgray',
                  )
    sns.stripplot(ax = ax,
                  data=df,
                  size=5,
                  jitter=True,
                 )
    
    
def index_dispersion(sdv):
    return calculate_sdv(sdv) / mean(sdv)



def flex_with_time_shifts(drug_type , path, folder_name,timepoints=60, splits=15, shifts=[1, 3, 5, 7, 10], small_worldness_calc = False):
    result_dict = defaultdict(list)
    all_flexibility = defaultdict(dict) #to store all the data of the flexibility to later save it to text
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
                    all_flexibility[shifts[i]][drug]=flex
                    
                    #calculate small worldness 
                    sw = []
                    if small_worldness_calc:
                        for split in split_roi[i]:
                            #create correlation matrix
                            brain_correlationMatrix = createCorrelationMatrix(split, "pearson")
                            #find the threshold
                            brain_threshold = findThreshold(brain_correlationMatrix,0.15)
                            #binarize
                            brain_binaryMatrix = binarize(brain_correlationMatrix, brain_threshold)

                            inner_sw = []
                            for j in range(5):
                                #i can't do this since the split is going to be size 90, but i have 150 brain data so it's not square
                                null2_correlationMatrix = randomizeCorrelationMatrix(brain_correlationMatrix)
                                null2_threshold = findThreshold(null2_correlationMatrix,0.15)
                                null2_binaryMatrix = binarize(null2_correlationMatrix,null2_threshold)

                                #calculate small worldness
                                small_worldness, normalized_ccoef, normalized_path_length  = calculate_small_worldness(brain_binaryMatrix, null2_binaryMatrix)
                                inner_sw.append(small_worldness)

                            sw.append(np.average(inner_sw))
                    #avg_sw = (np.average(sw))

                    result_dict[drug_base][shifts[i]].append((i, flex, sw))
              
                    
    #saving all the flexibility calculations on all 150 brain regions
    if small_worldness_calc:
        filename  = "results/all_flex/" + drug_type
        if not os.path.isdir(filename):
            os.makedirs(filename)

        for i in range(len(shifts)):
            l = [] #store the data
            for k,v in sorted(all_flexibility[shifts[i]].items(),key=lambda x:int(x[0].split("_")[1])): #sort it by the drug brain number id
                l.append(v)
            np.savetxt(filename+"/"+str(shifts[i])+".txt", np.array(l).T)
        
    #add code here to take the average over the 5 data brains you created
    total_drug = defaultdict(list)
    small_worldness = defaultdict(list)
    #take the average over each brain and store it
    for base,sfts in result_dict.items():
        for index, flex in sfts.items():
            flex_list = []
            sw_list = []
            for f in flex:
                flex_list.append(f[1])
                sw_list.append(f[2])
            total_drug[base].append((f[0], np.average(flex_list,axis = 0)))
            small_worldness[base].append((f[0], np.average(sw_list)))
            
    return total_drug,small_worldness,all_flexibility


def calculate_flexilibity_with_timeshifts(timepoints, splits, shifts, drugs_list, path, plot_ranges):
    """
    
    return flexibilty of each time shift for each drug in our data directory
    same parameters as flex_with_time_shifts
    
    """
    
    total_drug = defaultdict(list)
    
    sw = defaultdict(list)
    
    for drug in drugs_list:
        drug_dict, small_worldness, _ = flex_with_time_shifts(drug, path, timepoints, splits, shifts)
        total_drug[drug] = drug_dict
        sw[drug].append(small_worldness)
        print(drug)
        
    
        
    flexibility_calculations = defaultdict(tuple)
    
    fig = plt.figure(figsize = (30,50))
    

    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, shifts, drugs_list[i])
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC",plot_ranges[drugs_list[i]]["mean"])
        index+=1
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3",plot_ranges[drugs_list[i]]["sdv"])
        index+=1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV",plot_ranges[drugs_list[i]]["cov"])
        index+=1
    
        
    return flexibility_calculations, fig, sw
        

    
    
    
def calculate_flexilibity_with_timeshifts_and_save(timepoints, splits, shifts, drugs_list, path, plot_ranges, folder_name, small_worldness_calc =False):
    """
    
    return flexibilty of each time shift for each drug in our data directory
    same parameters as flex_with_time_shifts
    
    """
    
    total_drug = defaultdict(list)
    
    sw = defaultdict(list)
    
    for drug in drugs_list:
        drug_dict, small_worldness, _ = flex_with_time_shifts(drug, path, folder_name,timepoints, splits, shifts, small_worldness_calc)
        total_drug[drug] = drug_dict
        sw[drug].append(small_worldness)
        print(drug)
        
    
        
    flexibility_calculations = defaultdict(tuple)
    
    fig = plt.figure(figsize = (30,50))
    

    index = 1
    for i in range(len(drugs_list)):
        sdv, cov, fmean = tp_dict(total_drug, shifts, drugs_list[i])
        flexibility_calculations[drugs_list[i]] = (sdv, cov, fmean)
        box_plot(fmean, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03BC",plot_ranges[drugs_list[i]]["mean"])
        index+=1
        box_plot(sdv, (len(drugs_list), 3, index), fig, drugs_list[i], "\u03C3",plot_ranges[drugs_list[i]]["sdv"])
        index+=1
        box_plot(cov, (len(drugs_list), 3, index), fig, drugs_list[i], "CoV",plot_ranges[drugs_list[i]]["cov"])
        index+=1
        
        for key, value in fmean.items(): 
            filename  = folder_name+"/fmean/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)

        for key, value in cov.items(): 
            filename  = folder_name+"/cov/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)

        for key, value in sdv.items(): 
            filename  = folder_name+"/sdv/" + drugs_list[i]
            if not os.path.isdir(filename):
                os.makedirs(filename)
            np.savetxt(filename+"/"+str(key)+".txt", value)
    
        
    return flexibility_calculations, fig, sw

    

"""
Functions to Create a random time series with given bandwidth needed

""" 
    
#create random time series similar with a mean of 0 and a variance of about var
#do the same process as we did with the brain
#sigma * np.random.randn(...) + mu
#the signals need to each fall within aa frequency range 0.01 to 0.1


#functions taken from https://stackoverflow.com/questions/33933842/how-to-generate-noise-in-frequency-range-with-numpy
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


def getRandomTimeSeries(path, min_freq = 0.01, max_freq = 0.1, samples = 200, samplerate = 2, amt = 12, iterations = 5):
    """
    will return the given amount of time series that is generated randomly with the given bandwidth needed
    param: min_freq and max_freq is the bandwidth
    param: samples is the amount of data points you want in the time series
    param: samplerate is how often you do want the noise to be created
    param: amt is the amount of networks needed
    
    to simulate similar to our data we chosen these parameters
    
    """
    #make a directory for the time seriers
    
    timeseries = []
    for i in range(amt):
        for j in range(150):
            
            #create num iterations of the same brain content
            for k in range(iterations):
                roi = band_limited_noise(min_freq, max_freq, samples, samplerate)
                directory = path + "/RANDOM" + "_" + str(i+1) + "_" + str(k+1)
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                    
                #save the file in our directory
                filename = directory + "/roi" + str(j+2) + ".txt"
                np.savetxt(filename, roi)  
        
        timeseries.append(roi)

    return np.array(timeseries)
                
    
        
        
        
        
        

        
        

    