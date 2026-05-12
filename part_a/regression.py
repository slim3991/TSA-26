import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import statsmodels.api as sm


TIME_STEPS_PER_DAY = 12 * 24
TIME_STEPS_PER_WEEK = 12 * 24 * 7


def load_dataset(length: int = 10080, start: int = 27000) -> npt.NDArray:
    """Returns the median traffic in the dataset across all node pairs."""
    
    T = np.load("./raw_data/abiline_ten.npy")
    return np.median(T, axis=(0, 1))[start : start + length]


def test_linear_trend(data: npt.NDArray) -> None:
    t = np.arange(len(data), dtype=float)
    X = sm.add_constant(t)

    # Fit OLS, then correct SEs with Newey-West HAC estimator.
    # maxlags follows the common rule-of-thumb: floor(4 * (n/100)^(2/9))
    n = len(data)
    maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    ols_result = sm.OLS(data, X).fit()
    hac_result = ols_result.get_robustcov_results("HAC", maxlags=maxlags, use_correction=True)

    slope = hac_result.params[1]
    intercept = hac_result.params[0]
    se_slope = hac_result.bse[1]
    t_stat = hac_result.tvalues[1]
    p_value = hac_result.pvalues[1]
    ci = hac_result.conf_int(alpha=0.05)
    ci_low, ci_high = ci[1, 0], ci[1, 1]
    r_squared = ols_result.rsquared  # R² is a fit measure, not affected by SE correction

    print("=== Linear Trend Test with Newey-West HAC Standard Errors ===")
    print(f"  Newey-West maxlags: {maxlags}")
    print(f"  Slope:              {slope:.6f}  (change per time step)")
    print(f"  Intercept:          {intercept:.4f}")
    print(f"  HAC Std error:      {se_slope:.6f}")
    print(f"  t-statistic:        {t_stat:.4f}")
    print(f"  p-value:            {p_value:.4e}")
    print(f"  95% CI for slope:   [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  R²:                 {r_squared:.6f}")
    ci_excludes_zero = ci_low > 0 or ci_high < 0
    significant = p_value < 0.05
    print(f"\n  CI excludes zero: {ci_excludes_zero}")
    print(f"  Linear trend is {'STATISTICALLY SIGNIFICANT' if significant else 'NOT significant'} at α=0.05")

    print("\n=== Full HAC-Corrected OLS Summary ===")
    print(hac_result.summary())

    # --- Plot ---
    fitted = intercept + slope * t

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Full series with trend line
    axes[0].plot(t, data, color="steelblue", alpha=0.6, linewidth=0.8, label="Traffic (median)")
    axes[0].plot(t, fitted, color="red", linewidth=2, label=f"Linear trend (slope={slope:.4f})")
    axes[0].set_title(f"Linear Trend — p={p_value:.2e}, R²={r_squared:.4f}")
    axes[0].set_xlabel("Time Steps")
    axes[0].set_ylabel("Traffic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = data - fitted
    axes[1].plot(t, residuals, color="gray", alpha=0.6, linewidth=0.6, label="Residuals")
    axes[1].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_title("Residuals after detrending")
    axes[1].set_xlabel("Time Steps")
    axes[1].set_ylabel("Residual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    data = load_dataset()
    test_linear_trend(data)


if __name__ == "__main__":
    main()
