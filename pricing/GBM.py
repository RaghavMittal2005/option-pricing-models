import numpy as np
import matplotlib.pyplot as plt
from .utils import plot_paths

# --------------------------
# GBM Path Simulation
# --------------------------
def simulate_gbm_paths(S0, T, N, r, sigma, n_paths=10000, seed=42):
    """
    Simulates Geometric Brownian Motion (GBM) paths under Black-Scholes dynamics.
    """
    np.random.seed(seed)
    dt = T / N
    time = np.linspace(0, T, N+1)

    # Normal random variables
    Z = np.random.normal(0.0, 1.0, (n_paths, N))

    # Log process
    X = np.zeros((n_paths, N+1))
    X[:,0] = np.log(S0)

    for i in range(N):
        if n_paths > 1:  # normalize column for stability
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        X[:, i+1] = X[:, i] + (r - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*Z[:, i]

    S = np.exp(X)
    return time, X, S

# --------------------------
# Monte Carlo Option Pricing
# --------------------------
def monte_carlo_option_price(S0, K, T, r, sigma, N=100, n_paths=10000, option="call"):
    """
    Prices a European option using Monte Carlo with GBM paths.
    """
    _, _, S = simulate_gbm_paths(S0, T, N, r, sigma, n_paths)
    ST = S[:, -1]  # terminal stock prices

    if option == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    price = np.exp(-r*T) * np.mean(payoff)
    return price

# --------------------------
# Demo: Plot paths + Price
# --------------------------

def demo():
    S0 = 100
    K = 105
    T = 1
    r = 0.05
    sigma = 0.2
    N = 500
    n_paths = 25   # for plotting

    # Plot sample paths
    time, X, S = simulate_gbm_paths(S0, T, N, r, sigma, n_paths)
    plot_paths(time, S, title="GBM - Sample Stock Price Paths", ylabel="Stock Price")

    # Monte Carlo option pricing with many paths
    mc_call = monte_carlo_option_price(S0, K, T, r, sigma, N=100, n_paths=100000, option="call")
    mc_put  = monte_carlo_option_price(S0, K, T, r, sigma, N=100, n_paths=100000, option="put")

    print(f"European Call Option (MC): {mc_call:.4f}")
    print(f"European Put Option  (MC): {mc_put:.4f}")


