import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100          # Initial stock price
T = 1.0           # Time to maturity (1 year)
N = 100            # Number of time steps
r = 0.05          # Risk-free rate
sigma = 0.2       # Volatility

# Binomial parameters
dt = T / N
u = np.exp(sigma * np.sqrt(dt))
d = 1 / u
p = (np.exp(r * dt) - d) / (u - d)

# Simulate a few paths
n_paths = 50
paths = []

for _ in range(n_paths):
    steps = np.random.choice([u, d], size=N, p=[p, 1-p])
    price_path = [S0]
    for step in steps:
        price_path.append(price_path[-1] * step)
    paths.append(price_path)

# Plotting
for path in paths:
    plt.plot(range(N+1), path)
plt.title('Binomial Model: Sample Asset Price Paths')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.grid(True)
plt.show()
