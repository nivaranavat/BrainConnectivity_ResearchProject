import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

drug_names = {"SAL" : "Saline" , "COC" : "Cocaine", "MDPV" : "MDPV", "RANDOM": "Random"}

def plot_timeseries(timeseries, x=None, title="Time Series", ax=None, show=True, save_path=None):
    """
    Plot multiple time series.
    Args:
        timeseries: list or np.ndarray of shape (n_series, n_timepoints)
        x: x-axis values (optional)
        title: plot title
        ax: matplotlib axis (optional)
        show: whether to show the plot
        save_path: if provided, save the plot to this path
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    for y in timeseries:
        ax.plot(x if x is not None else np.arange(len(y)), y)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return ax

def plot_correlation_matrix(matrix, title="Correlation Matrix", save_path=None, show=True, ax=None, cmap="viridis", norm=None):
    """
    Plot a correlation matrix with colorbar.
    Args:
        matrix: 2D numpy array
        title: plot title
        save_path: if provided, save the plot to this path
        show: whether to show the plot
        ax: matplotlib axis (optional)
        cmap: colormap (default "viridis")
        norm: matplotlib.colors.Normalize object (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(matrix, cmap=cmap, norm=norm)
    ax.set_title(title)
    if save_path:
        plt.savefig(save_path)
    plt.colorbar(cax, ax=ax)
    if show:
        plt.show()
    return ax

def plot_flexibility_results(flexibility, title="Flexibility Results", save_path=None, show=True):
    """
    Plot flexibility results for each drug.
    Args:
        flexibility: dict of drug -> (sdv, cov, mean)
        title: plot title
        save_path: if provided, save the plot to this path
        show: whether to show the plot
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    drugs = list(flexibility.keys())
    sdv = [flexibility[d][0] for d in drugs]
    cov = [flexibility[d][1] for d in drugs]
    mean = [flexibility[d][2] for d in drugs]
    axs[0].bar(drugs, sdv)
    axs[0].set_title('Standard Deviation')
    axs[1].bar(drugs, cov)
    axs[1].set_title('Coefficient of Variation')
    axs[2].bar(drugs, mean)
    axs[2].set_title('Mean')
    fig.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    return fig, axs 


def plot_degree_histogram(matrix, title="Degree Histogram"):
    degrees = matrix.sum(axis=1)
    plt.hist(degrees)
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

def plot_matrix_difference(mat1, mat2, title="Matrix Difference"):
    plt.matshow(mat1 - mat2)
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_histogram(matrix, title="Matrix Histogram"):
    plt.hist(matrix)
    plt.title(title)
    plt.show()

    #graph out the boxplot of our data
def box_plot(tp_dict,subplot_index, figure, drug_type, y_value, y_range):
    df = pd.DataFrame([
        {"Timepoint": tp, "Value": val}
        for tp, values in tp_dict.items()
        for val in values
    ])

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

    sns.boxplot(ax=ax, data=df, x="Timepoint", y="Value", width=0.5, color='lightgray')
    sns.stripplot(ax=ax, data=df, x="Timepoint", y="Value", size=5, jitter=True)
    