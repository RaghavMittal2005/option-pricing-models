import numpy as np
from scipy.stats import norm
def bs_price(S0, K, T, r, sigma, option="call"):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)
    else: # put
        return norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S0


def bs_d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def bs_greeks(S, K, T, r, sigma, option="call"):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)

    delta = norm.cdf(d1) if option == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta_call = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    theta_put  = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    theta = theta_call if option == "call" else theta_put
    rho_call = K*T*np.exp(-r*T)*norm.cdf(d2)
    rho_put  = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    rho = rho_call if option == "call" else rho_put

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega/100,   # often quoted per 1% change in vol
        "Theta": theta/365, # per day
        "Rho": rho/100      # per 1% change in rate
    }

# Example
S0, K, T, r, sigma = 100, 105, 1, 0.05, 0.2
greeks_call = bs_greeks(S0, K, T, r, sigma, option="call")
greeks_put  = bs_greeks(S0, K, T, r, sigma, option="put")

print("Call Greeks:", greeks_call)
print("Put  Greeks:", greeks_put)
