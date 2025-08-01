import numpy as np
import matplotlib.pyplot as plt
from .utils import plot_paths


def simulate_binomial_paths(S0, T, N, r, sigma, n_paths=50):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    if not (0 <= p <= 1):
        raise ValueError(f"Invalid risk-neutral probability p={p:.4f}. "
                         f"Check parameters: d={d:.4f}, exp(r dt)={np.exp(r*dt):.4f}, u={u:.4f}")

    paths = []
    for _ in range(n_paths):
        steps = np.random.choice([u, d], size=N, p=[p, 1-p])
        price_path = [S0]
        for step in steps:
            price_path.append(price_path[-1] * step)
        paths.append(price_path)
    time = np.arange(N+1)
    paths = np.array(paths)
    plot_paths(time, paths, title="Binomial Model - Sample Stock Price Paths", ylabel="Stock Price")
    return paths


def binomial_option_price(S0, K, T, r, sigma, N, option="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # stock prices at maturity
    ST = np.array([S0 * (u**j) * (d**(N-j)) for j in range(N+1)])
    
    # option payoff at maturity
    if option == "call":
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)

    # backward induction
    for i in range(N-1, -1, -1):
        C = np.exp(-r*dt) * (p * C[1:] + (1-p) * C[:-1])
    
    return C[0]


