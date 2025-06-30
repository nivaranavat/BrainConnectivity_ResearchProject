# Auto-generated from notebook

def cell_1():
    import utils 
    
     #constant for our research purposes
    #needs to be the path to where the data is
    path = "/Users/nivaranavat/UCI Research/Data"

def cell_2():
    # #create our random time series
    #utils.getRandomTimeSeries(path, min_freq = 0.01, max_freq = 0.1, samples = 200, samplerate =2, amt = 12)

def cell_3():
    # #phase scramble the drugs that we have
    # #calculate the small worldness of each brain observed with the scrambled brain
    # #need to figure out a threshold 
    
    # drug_list = ("SAL", "COC", "MDPV")
    # utils.phaseScrambled(drug_list, path, filesize = 200, iterations = 5, threshold = 0.15)

def cell_4():
    #calculate flexibility of each every data set
    drug_list = ("RANDOM","SAL", "OrgSCRSAL","SCRSAL","COC","OrgSCRCOC","SCRCOC", "MDPV","OrgSCRMDPV","SCRMDPV")
    plot_ranges = {"RANDOM" : {"mean" : [0,0.11], "cov" : [0.15, 2] , "sdv": [0,0.04] } ,
                  "SAL" : {"mean" : [0,0.11], "cov" : [0.15, 2.6] , "sdv": [0,0.04] }, 
                  "OrgSCRSAL" : {"mean" : [0,0.125], "cov" : [0.15, 2] , "sdv": [0,0.04] },
                  "SCRSAL" : {"mean" : [0,0.11], "cov" : [0.15, 2] , "sdv": [0,0.04] },
                  "COC": {"mean" : [0,0.11], "cov" : [0.15, 2.6] , "sdv": [0,0.04] },
                   "OrgSCRCOC": {"mean" : [0,0.125], "cov" : [0.15, 2] , "sdv": [0,0.04] },
                  "SCRCOC": {"mean" : [0,0.15], "cov" : [0.15, 2] , "sdv": [0,0.04] },
                  "MDPV": {"mean" : [0,0.11], "cov" : [0.15, 2.6] , "sdv": [0,0.04] },
                   "OrgSCRMDPV": {"mean" : [0,0.125], "cov" : [0.15, 2] , "sdv": [0,0.04] },
                  "SCRMDPV": {"mean" : [0,0.11], "cov" : [0.15, 2] , "sdv": [0,0.04] }}
    
    timepoints = [30,60,90,120,150]
    folder_name = "results/timewindows[30,60,90,120,150]"
    flexibility, fig = utils.calculate_flexibility(path, drug_list, timepoints, plot_ranges,folder_name)
    fig.suptitle(f"Flexibility's Standard Deviation, Coefficient of Variation, and Mean for each Drug with time windows {timepoints}, splits of 50 and shifts of 1")
    fig.savefig("flexibility__with_timepoints[30,60,90,120,150].png")

def cell_5():
    #calculate flexibility of each every data set we have with the time shifts
    drug_list = ("RANDOM","SAL", "OrgSCRSAL","SCRSAL","COC","OrgSCRCOC","SCRCOC", "MDPV","OrgSCRMDPV","SCRMDPV")
    plot_ranges = {"RANDOM" : {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.35] } ,
                  "SAL" : {"sdv" : [0,0.15], "cov" : [0.15, 5.25] , "mean": [0,0.35] }, 
                   "OrgSCRSAL" : {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.4] },
                  "SCRSAL" : {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.35] },
                  "COC": {"sdv" : [0,0.15], "cov" : [0.15, 5.25] , "mean": [0,0.35] },
                   "OrgSCRCOC": {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.4] },
                  "SCRCOC": {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.35] },
                  "MDPV": {"sdv" : [0,0.15], "cov" : [0.15, 5.25] , "mean": [0,0.35] },
                   "OrgSCRMDPV": {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.4] },
                  "SCRMDPV": {"sdv" : [0,0.15], "cov" : [0.15, 2.5] , "mean": [0,0.35] }}
    
    timepoints = 90
    splits = 12
    shifts = [1,3,5,7,10]
    folder_name = "results/timeshifts[1,3,5,7,10]"
    flexibility,fig, small_worldness = utils.calculate_flexilibity_with_timeshifts_and_save(timepoints, splits, shifts, drug_list, path, plot_ranges, folder_name)
    fig.suptitle(f"Flexibility's Standard Deviation, Coefficient of Variation, and Mean for each Drug with time windows {timepoints} splits of {splits} and shifts of {shifts}")
    fig.savefig("flexibility_with_timewindow90_timeshifts[1,3,5,7,10].png")

def cell_6():
    #calculate flexibility of each every data set we have with the time shifts
    drug_list = ("RANDOM","SAL", "OrgSCRSAL","SCRSAL","COC","OrgSCRCOC","SCRCOC", "MDPV","OrgSCRMDPV","SCRMDPV")
    plot_ranges = {"RANDOM" : {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] } ,
                  "SAL" : {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] }, 
                   "OrgSCRSAL" : {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                  "SCRSAL" : {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                  "COC": {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                   "OrgSCRCOC": {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                  "SCRCOC": {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                  "MDPV": {"sdv" : [0.03,0.1], "cov" : [0, 1] , "mean": [0.07,0.25] },
                   "OrgSCRMDPV": {"sdv" : [0.03,0.1], "cov" : [0,1] , "mean": [0.07,0.25] },
                  "SCRMDPV": {"sdv" : [0.03,0.1], "cov" : [0,1] , "mean": [0.07,0.25] }}
    
    timepoints = 60
    splits = 28
    shifts = [5]
    folder_name = "results/timeshift5_splits28"
    #need to take the small worldness average here
    flexibility,fig,small_worldness = utils.calculate_flexilibity_with_timeshifts_and_save(timepoints, splits, shifts, drug_list, path,plot_ranges,folder_name, small_worldness_calc = True)
    fig.suptitle(f"Flexibility's Standard Deviation, Coefficient of Variation, and Mean for each Drug with time windows {timepoints}, splits of {splits} and shifts of {shifts}")
    fig.savefig("flexibility_with_timewindow60_timeshifts5_28splits.png")

def cell_7():
    print(f"Small Worldness with timeshifts {shifts} calculated with {timepoints} size time window")
        
    import os
        
    if not os.path.isdir("small_worldness"):
        os.makedirs("small_worldness")
    for drug in drug_list:
        filename  = "small_worldness/" + drug + ".txt"
        with open(filename,"w") as f:
                for key, value in small_worldness[drug][0].items(): 
                    f.write('%s:%s\n' % ( key,value[0][1]))

def cell_8():
    from matplotlib import pyplot as plt
    #draw out our last plot that is going to cover all the values calculated for the specified parameters
    drug_sdv = {}
    drug_cov = {}
    drug_mean = {}
    
    drug_names = {"SAL" : "Saline" , "COC" : "Cocaine", "MDPV" : "MDPV", "RANDOM": "Random"} #mapping of the abbreivated name to full name of drug
    
    for drug, flex in flexibility.items():
        if drug in drug_names:
            name = drug_names[drug]
        if "SCR" in drug:
            name = "\u03BB'" + drug_names[drug.split("SCR")[1]]
        if "OrgSCR" in drug:
            name = "\u03BB''" + drug_names[drug.split("SCR")[1]]
        drug_sdv[name] = flex[0][5]
        drug_cov[name] = flex[1][5]
        drug_mean[name] = flex[2][5]
    
    fig = plt.figure(figsize = (15,15))
    utils.box_plot(drug_sdv, (1,1,1), fig, "", "\u03C3", [0.02,0.1])
    fig.savefig("sdv_window60_shift5.png")
        
    
    fig = plt.figure(figsize = (15,15))
    utils.box_plot(drug_cov, (1,1,1), fig, "", "CoV", [0, 1])
    fig.savefig("cov_window60_shift5.png")
    
    
    fig = plt.figure(figsize = (15,15))
    utils.box_plot(drug_mean, (1,1,1), fig, "", "\u03BC", [0,0.25])
    fig.savefig("mean_window60_shift5.png")
        

def cell_9():


def cell_10():

