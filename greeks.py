import numpy as np
from pricing.binomial import simulate_binomial_paths,binomial_option_price
def compute_greeks(pricer, S0, K, T, r, sigma, h=1e-4, option="call", **kwargs):
    """
    Compute Greeks via finite difference for any pricing model.
    
    Parameters:
        pricer  : function(S0, K, T, r, sigma, option=..., **kwargs) -> price
        S0,K,T,r,sigma : option params
        h       : small bump for finite differences
        option  : "call" or "put"
        kwargs  : extra params (for Merton: rate, muJ, sigmaJ; for MC: N, n_paths, etc.)
    """

    # Base price
    V = pricer(S0, K, T, r, sigma, option=option, **kwargs)

    # Delta
    V_up   = pricer(S0+h, K, T, r, sigma, option=option, **kwargs)
    V_down = pricer(S0-h, K, T, r, sigma, option=option, **kwargs)
    delta = (V_up - V_down) / (2*h)

    # Gamma
    gamma = (V_up - 2*V + V_down) / (h**2)

    # Vega
    V_up   = pricer(S0, K, T, r, sigma+h, option=option, **kwargs)
    V_down = pricer(S0, K, T, r, sigma-h, option=option, **kwargs)
    vega = (V_up - V_down) / (2*h)

    # Theta (reduce/increase maturity)
    if T-h > 0:
        V_up   = pricer(S0, K, T+h, r, sigma, option=option, **kwargs)
        V_down = pricer(S0, K, T-h, r, sigma, option=option, **kwargs)
        theta = (V_down - V_up) / (2*h)
    else:
        theta = np.nan  # undefined if T-h <= 0

    # Rho
    V_up   = pricer(S0, K, T, r+h, sigma, option=option, **kwargs)
    V_down = pricer(S0, K, T, r-h, sigma, option=option, **kwargs)
    rho = (V_up - V_down) / (2*h)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta": theta,
        "Rho": rho
    }

greeks_binomial = compute_greeks(binomial_option_price, 100, 105, 1, 0.05, 0.2, N=200, option="call")
print("Binomial Greeks:", greeks_binomial)
