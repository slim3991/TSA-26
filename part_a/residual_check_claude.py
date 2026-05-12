"""
Residual (Innovation) Diagnostics for the fitted SARIMAX model.

Checks that the fitted innovations U_hat_t = Z_t - Z_hat_t behave like
white noise, which is required for the model to be valid and for the
bootstrap confidence intervals to be meaningful.

Three core checks (from the lecture):
  1. Time plot of U_hat  — should look structureless
  2. Sample ACF of U_hat — should be negligible for all lags h >= 1
  3. Sample ACF of U_hat^2 — checks for remaining volatility clustering

Additional checks:
  4. Histogram + Q-Q plot — assesses Gaussian assumption
  5. Ljung-Box test       — formal statistical test for white noise
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from main_anton import load_dataset, make_train_test_split, TIME_STEPS_PER_WEEK

# ── Configuration ────────────────────────────────────────────────────────────
MAX_LAGS = 60       # Number of ACF lags to display
LB_LAGS  = 20       # Number of lags for the Ljung-Box test
AR, MA   = 4, 2     # Must match what was used in main_anton.py
ALPHA    = 0.05     # Significance level for Ljung-Box
# ─────────────────────────────────────────────────────────────────────────────


def fit_model_for_diagnostics(train_data: np.ndarray, ar: int, ma: int):
    """
    Fast equivalent of fit_integrated_model for residual diagnostics only.

    Applies seasonal (period=2016) and regular differencing manually before
    fitting ARIMA(ar,0,ma). This avoids the SARIMAX Kalman state vector of
    size ~2016 that makes the native seasonal_order approach extremely slow.
    The innovations from this model are identical to those of the full SARIMAX.
    """
    seasonal_diff = train_data[TIME_STEPS_PER_WEEK:] - train_data[:-TIME_STEPS_PER_WEEK]
    W = np.diff(seasonal_diff)
    model = SARIMAX(
        W,
        order=(ar, 0, ma),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def extract_innovations(results) -> np.ndarray:
    """
    Returns the fitted innovations U_hat_t = Z_t - Z_hat_t.

    statsmodels stores these as results.resid. By Proposition 4.14 of the
    lecture, for large n these converge to the true white noise shocks W_t,
    so their empirical distribution approximates the noise distribution.
    """
    return np.array(results.resid)


def plot_time_series(u_hat: np.ndarray) -> None:
    """
    Check 1: Time plot.

    The innovations should look like structureless noise — no trend,
    no seasonality, no changing variance. Any visible pattern means
    the model has not captured all dependence in the data.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(u_hat, color="steelblue", linewidth=0.6, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Check 1 — Time plot of innovations $\\hat{U}_t$\n"
                 "Expected: structureless noise around zero, constant spread")
    ax.set_xlabel("Time step")
    ax.set_ylabel("$\\hat{U}_t$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acf_levels(u_hat: np.ndarray) -> None:
    """
    Check 2: Sample ACF of U_hat.

    For white noise, rho(h) = 0 for all h >= 1 (Definition 1.3). So all
    bars should fall inside the 95% confidence bands (blue shaded region).
    Significant spikes indicate remaining autocorrelation — the model is
    missing structure that should be captured.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(u_hat, lags=MAX_LAGS, ax=ax, color="steelblue", alpha=0.05)
    ax.set_title("Check 2 — Sample ACF of innovations $\\hat{U}_t$\n"
                 "Expected: all lags inside the 95% bands (white noise)")
    ax.set_xlabel("Lag $h$")
    ax.set_ylabel("$\\hat{\\rho}(h)$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acf_squared(u_hat: np.ndarray) -> None:
    """
    Check 3: Sample ACF of U_hat^2.

    Even if U_hat looks uncorrelated, the squared series can reveal
    volatility clustering — periods where large shocks are followed by
    more large shocks (an ARCH/GARCH effect, Section 6.2-6.3 of lecture).
    If this ACF has significant spikes, a stochastic volatility model
    would be more appropriate than assuming iid innovations.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(u_hat**2, lags=MAX_LAGS, ax=ax, color="darkorange", alpha=0.05)
    ax.set_title("Check 3 — Sample ACF of squared innovations $\\hat{U}_t^2$\n"
                 "Expected: all lags inside the 95% bands (no volatility clustering)")
    ax.set_xlabel("Lag $h$")
    ax.set_ylabel("$\\hat{\\rho}_{U^2}(h)$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_gaussianity(u_hat: np.ndarray) -> None:
    """
    Check 4: Histogram and Q-Q plot.

    The Gaussian assumption underlies the analytical confidence intervals
    (the 1.96 * sigma_h formula from Theorem 4.15). If the residuals are
    heavy-tailed or skewed, the bootstrap resampling approach is preferable
    over assuming N(0, sigma^2) for the Monte Carlo draws.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Histogram with fitted Gaussian overlay
    ax = axes[0]
    u_std = (u_hat - u_hat.mean()) / u_hat.std()
    ax.hist(u_std, bins=60, density=True, color="steelblue",
            alpha=0.6, label="Standardised $\\hat{U}_t$")
    x = np.linspace(u_std.min(), u_std.max(), 300)
    ax.plot(x, stats.norm.pdf(x), color="red", linewidth=2, label="N(0,1)")
    ax.set_title("Check 4a — Histogram of standardised innovations\n"
                 "Expected: close to N(0,1) if Gaussian assumption holds")
    ax.set_xlabel("Standardised value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-Q plot
    ax = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(u_hat, dist="norm")
    ax.scatter(osm, osr, color="steelblue", s=4, alpha=0.5, label="Innovations")
    ax.plot(osm, slope * np.array(osm) + intercept,
            color="red", linewidth=2, label="Normal reference line")
    ax.set_title("Check 4b — Q-Q plot\n"
                 "Expected: points on the line if Gaussian; tails reveal heavy tails")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_ljung_box(u_hat: np.ndarray) -> None:
    """
    Check 5: Ljung-Box test for remaining autocorrelation.

    Formal test of H0: rho(1) = rho(2) = ... = rho(LB_LAGS) = 0.
    A p-value above ALPHA means we cannot reject the white noise hypothesis
    at that lag — which is what we want. Rejections (low p-values) indicate
    the model is mis-specified.
    """
    lb = acorr_ljungbox(u_hat, lags=LB_LAGS, return_df=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(lb.index, lb["lb_pvalue"], color="steelblue", alpha=0.7)
    ax.axhline(ALPHA, color="red", linestyle="--",
               linewidth=1.5, label=f"α = {ALPHA}")
    ax.set_title("Check 5 — Ljung-Box p-values\n"
                 "Expected: all bars ABOVE the red line (fail to reject white noise)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("p-value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nLjung-Box Test Results:")
    print(f"{'Lag':>5}  {'Statistic':>12}  {'p-value':>10}  {'White noise?':>14}")
    print("-" * 50)
    for lag, row in lb.iterrows():
        verdict = "Yes" if row["lb_pvalue"] > ALPHA else "NO — reject"
        print(f"{lag:>5}  {row['lb_stat']:>12.4f}  {row['lb_pvalue']:>10.4f}  {verdict:>14}")


def print_summary(u_hat: np.ndarray) -> None:
    """Prints basic summary statistics of the innovations."""
    print("\n── Innovation Summary Statistics ───────────────────────────")
    print(f"  n            : {len(u_hat)}")
    print(f"  Mean         : {u_hat.mean():.6f}  (expected ≈ 0)")
    print(f"  Std (σ̂)      : {u_hat.std():.6f}  (this is your estimated σ)")
    print(f"  σ̂²           : {u_hat.std()**2:.6f}  (use for MC draws)")
    print(f"  Skewness     : {float(((u_hat - u_hat.mean())**3).mean() / u_hat.std()**3):.4f}  (expected ≈ 0 if Gaussian)")
    print(f"  Excess kurt. : {float(((u_hat - u_hat.mean())**4).mean() / u_hat.std()**4 - 3):.4f}  (expected ≈ 0 if Gaussian)")
    print("─────────────────────────────────────────────────────────────\n")


def main():
    # Load data and fit model (mirrors main_anton.py)
    print("Loading data and fitting model...")
    Tp = load_dataset()
    train_raw, _ = make_train_test_split(Tp, 0.9)
    results = fit_model_for_diagnostics(train_raw, AR, MA)

    # Extract innovations
    u_hat = extract_innovations(results)
    print(f"Extracted {len(u_hat)} innovations from fitted model.\n")

    print_summary(u_hat)

    # Run all checks
    plot_time_series(u_hat)
    plot_acf_levels(u_hat)
    plot_acf_squared(u_hat)
    plot_gaussianity(u_hat)
    run_ljung_box(u_hat)

    print("\nInterpretation guide:")
    print("  Check 2 clean, Check 3 clean → bootstrap OR Gaussian MC both valid")
    print("  Check 2 clean, Check 3 dirty → volatility clustering; prefer bootstrap")
    print("  Check 2 dirty               → model is mis-specified; revisit ARMA order")
    print("  Check 4 heavy tails         → prefer bootstrap over Gaussian MC draws")


if __name__ == "__main__":
    main()