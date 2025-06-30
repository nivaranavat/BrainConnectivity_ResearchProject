import matplotlib.pyplot as plt
import numpy as np

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

def plot_correlation_matrix(matrix, title="Correlation Matrix", save_path=None, show=True):
    """
    Plot a correlation matrix with colorbar.
    Args:
        matrix: 2D numpy array
        title: plot title
        save_path: if provided, save the plot to this path
        show: whether to show the plot
    """
    plt.figure(figsize=(6, 6))
    plt.matshow(matrix)
    plt.title(title)
    plt.colorbar()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

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