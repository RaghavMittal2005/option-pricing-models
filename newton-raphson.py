import numpy as np
from scipy.stats import norm

# ---------- Black-Scholes (for testing) ----------
def bs_price(S0, K, T, r, sigma, option="call"):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option == "call":
        return norm.cdf(d1)*S0 - norm.cdf(d2)*K*np.exp(-r*T)
    else: # put
        return norm.cdf(-d2)*K*np.exp(-r*T) - norm.cdf(-d1)*S0

def bs_vega(S0, K, T, r, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S0 * norm.pdf(d1) * np.sqrt(T)

# ---------- Generic implied volatility solver ----------
def implied_volatility(pricer, S0, K, T, r, market_price,
                       option="call", sigma_init=0.2, tol=1e-8, max_iter=100,
                       analytic_vega=None, **kwargs):
    """
    Implied volatility via Newton-Raphson.

    Parameters:
        pricer        : function(S0,K,T,r,sigma,option=...,**kwargs) -> price
        S0,K,T,r      : option parameters
        market_price  : observed option price
        option        : "call" or "put"
        sigma_init    : starting volatility
        tol           : tolerance
        max_iter      : maximum iterations
        analytic_vega : function(S0,K,T,r,sigma) -> vega (optional)
        kwargs        : extra args for model (e.g. N, n_paths, rate, muJ, sigmaJ)
    """
    sigma = sigma_init
    for i in range(max_iter):
        price = pricer(S0, K, T, r, sigma, option=option, **kwargs)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        # Use analytic vega if available, else numerical FD
        if analytic_vega is not None:
            vega = analytic_vega(S0, K, T, r, sigma)
        else:
            h = 1e-4
            price_up = pricer(S0, K, T, r, sigma+h, option=option, **kwargs)
            price_down = pricer(S0, K, T, r, sigma-h, option=option, **kwargs)
            vega = (price_up - price_down) / (2*h)

        # Guard against zero vega
        if vega < 1e-8:
            raise ValueError("Vega too small — Newton method fails.")

        sigma -= diff / vega

    raise ValueError("Implied volatility did not converge")
