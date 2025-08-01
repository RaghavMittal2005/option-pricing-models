import matplotlib.pyplot as plt
import numpy as np

def plot_paths(time, paths, title="Simulated Paths", xlabel="Time", ylabel="Value",
               n_paths_to_plot=25, figsize=(8,5)):
    """
    Generic path plotting function for option pricing models.

    Parameters:
        time            : array of time steps
        paths           : 2D numpy array [n_paths x len(time)]
        title           : plot title
        xlabel, ylabel  : axis labels
        n_paths_to_plot : number of paths to plot (default 25)
        figsize         : figure size
    """
    plt.figure(figsize=figsize)
    n_paths = min(n_paths_to_plot, paths.shape[0])

    for i in range(n_paths):
        plt.plot(time, paths[i], lw=1, alpha=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def set_seed(seed=42):
    """
    Fix random seed for reproducibility.
    """
    np.random.seed(seed)

def martingale(S,r,T):
    M = lambda r,t: np.exp(r*t)
    ES = np.mean(S[:,-1])
    print(ES)
    ESM = np.mean(S[:,-1]/M(r,T))
    print(ESM)

