import numpy as np
import matplotlib.pyplot as plt
from .utils import plot_paths
# --------------------------
# Merton Jump-Diffusion Path Simulation
# --------------------------
def simulate_merton_paths(S0, T, N, r, sigma, n_paths=10000, rate=1.0, muJ=0.0, sigmaJ=0.7, seed=42):
    np.random.seed(seed)
    dt = T / N
    time = np.linspace(0, T, N+1)

    # Brownian motion increments
    Z = np.random.normal(0.0, 1.0, (n_paths, N))

    # Jump process
    J = np.random.normal(muJ, sigmaJ, (n_paths, N))       # jump sizes
    N_jump = np.random.poisson(rate * dt, (n_paths, N))   # number of jumps

    Ej = np.exp(muJ + 0.5 * sigmaJ**2)  # expectation adjustment

    X = np.zeros((n_paths, N+1))
    X[:,0] = np.log(S0)

    for i in range(N):
        if n_paths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        drift = (r - 0.5*sigma**2 - rate*(Ej-1)) * dt
        diffusion = sigma * np.sqrt(dt) * Z[:, i]
        jump = J[:, i] * N_jump[:, i]
        X[:, i+1] = X[:, i] + drift + diffusion + jump

    S = np.exp(X)
    return time, X, S

# --------------------------
# Monte Carlo Option Pricing under Merton
# --------------------------
def merton_option_price(S0, K, T, r, sigma, N=100, n_paths=100000,
                        rate=1.0, muJ=0.0, sigmaJ=0.7, option="call"):
    """
    Monte Carlo European option pricing under Merton jump-diffusion.
    """
    _, _, S = simulate_merton_paths(S0, T, N, r, sigma, n_paths, rate, muJ, sigmaJ)
    ST = S[:, -1]

    if option == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    price = np.exp(-r*T) * np.mean(payoff)
    return price

# --------------------------
# Demo
# --------------------------
def demo_merton():
    S0 = 100
    K = 105
    T = 1
    r = 0.05
    sigma = 0.2
    N = 500
    n_paths_plot = 25
    n_paths_mc = 100000

    # Plot a few paths
    time, _, S = simulate_merton_paths(S0, T, N, r, sigma, n_paths_plot, rate=1, muJ=0, sigmaJ=0.7)
    plot_paths(time, S, title="Merton Jump Diffusion - Stock Price Paths", ylabel="Stock Price")

    # Monte Carlo pricing
    call_price = merton_option_price(S0, K, T, r, sigma, N, n_paths_mc, rate=1, muJ=0, sigmaJ=0.7, option="call")
    put_price  = merton_option_price(S0, K, T, r, sigma, N, n_paths_mc, rate=1, muJ=0, sigmaJ=0.7, option="put")

    print(f"European Call Option (Merton MC): {call_price:.4f}")
    print(f"European Put Option  (Merton MC): {put_price:.4f}")

# Run demo
demo_merton()
