import numpy as np
from scipy.stats import norm
from .binomial import binomial_option_price,simulate_binomial_paths
import inspect

def implied_volatility(pricer, S0, K, T, r, market_price,
                       option="call", sigma_init=0.2, tol=1e-8, max_iter=100,
                       analytic_vega=None, **kwargs):
    import inspect
    sig = inspect.signature(pricer).parameters

    def safe_pricer(S0, K, T, r, sigma, option="call", **kwargs):
        valid_kwargs = {k: v for k, v in kwargs.items() if k in sig}
        return pricer(S0, K, T, r, sigma, option=option, **valid_kwargs)

    # Try Newton-Raphson
    sigma = sigma_init
    for i in range(max_iter):
        price = safe_pricer(S0, K, T, r, sigma, option=option, **kwargs)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        if analytic_vega is not None:
            vega = analytic_vega(S0, K, T, r, sigma)
        else:
            h = 1e-4
            price_up = safe_pricer(S0, K, T, r, sigma+h, option=option, **kwargs)
            price_down = safe_pricer(S0, K, T, r, sigma-h, option=option, **kwargs)
            vega = (price_up - price_down) / (2*h)

        # If Vega is too small, break and fallback
        if abs(vega) < 1e-8:
            break

        sigma -= diff / vega

    # --- Fallback: Bisection ---
    low, high = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5*(low+high)
        price_mid = safe_pricer(S0, K, T, r, mid, option=option, **kwargs)
        if price_mid > market_price:
            high = mid
        else:
            low = mid
        if abs(price_mid - market_price) < tol:
            return mid

    raise ValueError("Implied volatility did not converge")

print(implied_volatility(binomial_option_price, 100, 105, 1, 0.05, 0.2, N=200, option="call"))