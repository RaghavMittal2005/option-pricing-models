import streamlit as st
import numpy as np
from pricing.black_scholes import bs_price, bs_greeks
from pricing.binomial import binomial_option_price, simulate_binomial_paths
from pricing.GBM import monte_carlo_option_price, simulate_gbm_paths
from pricing.Merton import merton_option_price, simulate_merton_paths
from pricing.utils import plot_paths
from pricing.newton_raphson import implied_volatility  # your generalized IV solver

# --- Sidebar ---
st.sidebar.header("Option Parameters")
model = st.sidebar.selectbox("Pricing Model", ["Black-Scholes", "Binomial", "GBM Monte Carlo", "Merton Jump Diffusion"])
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T)", value=1.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2)

# Extra params
N = st.sidebar.slider("Steps", 50, 500, 200)
n_paths = st.sidebar.slider("Number of Paths (MC)", 10, 2000, 200)

# --- Main Output ---
st.title("📈 Option Pricing Models")

if model == "Black-Scholes":
    price = bs_price(S0, K, T, r, sigma, option=option_type)
    greeks = bs_greeks(S0, K, T, r, sigma, option=option_type)

    st.subheader("Black-Scholes Results")
    st.write("Price:", price)
    st.json(greeks)

elif model == "Binomial":
    price = binomial_option_price(S0, K, T, r, sigma, N, option=option_type)
    paths = simulate_binomial_paths(S0, T, N, r, sigma, n_paths)

    st.subheader("Binomial Results")
    st.write("Price:", price)
    st.line_chart(np.array(paths).T)

elif model == "GBM Monte Carlo":
    price = monte_carlo_option_price(S0, K, T, r, sigma, N, n_paths, option=option_type)
    time, _, S = simulate_gbm_paths(S0, T, N, r, sigma, n_paths)

    st.subheader("GBM Monte Carlo Results")
    st.write("Price:", price)
    st.line_chart(S.T)

elif model == "Merton Jump Diffusion":
    muJ = st.sidebar.number_input("Jump Mean (μJ)", value=0.0)
    sigmaJ = st.sidebar.number_input("Jump Vol (σJ)", value=0.3)
    rate = st.sidebar.number_input("Jump Intensity (λ)", value=1.0)

    price = merton_option_price(S0, K, T, r, sigma, N, n_paths, rate, muJ, sigmaJ, option=option_type)
    time, _, S = simulate_merton_paths(S0, T, N, r, sigma, n_paths, rate, muJ, sigmaJ)

    st.subheader("Merton Jump Diffusion Results")
    st.write("Price:", price)
    st.line_chart(S.T)

# --- Implied Volatility ---
st.sidebar.subheader("Implied Volatility")
market_price = st.sidebar.number_input("Market Option Price", value=0.0)
if market_price > 0:
    try:
        iv = implied_volatility(
            bs_price if model == "Black-Scholes" else
            binomial_option_price if model == "Binomial" else
            monte_carlo_option_price if model == "GBM Monte Carlo" else
            merton_option_price,
            S0, K, T, r, market_price, option=option_type,
            sigma_init=0.2, N=N, n_paths=n_paths
        )
        st.write("Implied Volatility:", iv)
    except Exception as e:
        st.error(f"IV calculation failed: {e}")
