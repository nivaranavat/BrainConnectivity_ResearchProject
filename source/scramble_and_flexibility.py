from source.utils.plot_utils import plot_flexibility_results
from source.utils import calculate_flexibility
import matplotlib.pyplot as plt

# Main function tying together the workflow
def main(
    data_path,
    drug_list,
    timepoints,
    plot_ranges,
    folder_name,
    plot=True,
    save_plots=False
):
    """
    Main workflow for scramble and flexibility analysis.
    Args:
        data_path: path to data
        drug_list: list of drugs
        timepoints: list of timepoints
        plot_ranges: dict of plot ranges
        folder_name: folder to save results
        plot: whether to show plots
        save_plots: whether to save plots
    Returns:
        flexibility results
    """
    flexibility, fig = calculate_flexibility(data_path, drug_list, timepoints, plot_ranges, folder_name)
    if plot:
        fig.suptitle(f"Flexibility's Standard Deviation, Coefficient of Variation, and Mean for each Drug with time windows {timepoints}")
        if save_plots:
            fig.savefig(f"{folder_name}/flexibility_plot.png")
        plt.show()
    return flexibility
