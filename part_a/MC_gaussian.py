"""
Gaussian Monte Carlo confidence intervals for future predictions.

Identical pipeline to MC_bootstrap.py but instead of resampling residuals,
innovations are drawn from N(0, σ̂²) where σ̂ is estimated from the fitted
residuals. Valid when the residual diagnostics show no heavy tails or skew.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from main_anton import load_dataset, make_train_test_split, TIME_STEPS_PER_WEEK

# ── Configuration ────────────────────────────────────────────────────────────
AR          = 4
MA          = 2
N_SIMS      = 10000
ALPHA       = 0.05
TRAIN_RATIO = 0.8
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────


def fit_model(train_data: np.ndarray, ar: int, ma: int):
    """
    Aim is to make the series sationary and fit ARMA
    """
    T = train_data.copy()
    seasonal_diff = T[TIME_STEPS_PER_WEEK:] - T[:-TIME_STEPS_PER_WEEK]
    T = np.diff(seasonal_diff, 1)
    # seasonal lag
    model = SARIMAX(
        T,
        order=(ar, 0, ma),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results, seasonal_diff


def gaussian_forecast(
    results,
    seasonal_diff: np.ndarray,
    train_data: np.ndarray,
    steps: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns an (n_sims, steps) array of Gaussian MC forecast paths
    in the original (un-differenced) domain.

    Innovations are drawn i.i.d. from N(0, σ̂²) rather than resampled,
    assuming the true noise distribution is Gaussian.
    """
    ar_params = results.arparams
    ma_params = results.maparams
    p, q      = len(ar_params), len(ma_params)

    residuals = np.asarray(results.resid)
    #breakpoint()
    sigma     = residuals.std()

    W        = np.diff(seasonal_diff)
    W_seed   = W[-(max(p, 1)):]
    eps_seed = residuals[-(max(q, 1)):]
    last_d   = seasonal_diff[-1]

    T             = len(train_data)
    seasonal_base = train_data[T - TIME_STEPS_PER_WEEK : T - TIME_STEPS_PER_WEEK + steps]

    paths = np.empty((n_sims, steps))

    for s in range(n_sims):
        eps_future = rng.normal(0, sigma, size=steps)

        W_hist   = list(W_seed)
        eps_hist = list(eps_seed)
        d        = last_d
        d_path   = np.empty(steps)

        for h in range(steps):
            ar_part = float(sum(ar_params[i] * W_hist[-(i + 1)] for i in range(p)))
            ma_part = float(sum(ma_params[j] * eps_hist[-(j + 1)] for j in range(q)))
            W_new   = ar_part + eps_future[h] + ma_part
            d      += W_new
            d_path[h] = d
            W_hist.append(W_new)
            eps_hist.append(eps_future[h])

        paths[s] = d_path + seasonal_base

    return paths


def plot_results(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    paths: np.ndarray,
    alpha: float,
) -> tuple:
    lower  = np.percentile(paths, 100 * alpha / 2,       axis=0)
    upper  = np.percentile(paths, 100 * (1 - alpha / 2), axis=0)
    median = np.median(paths, axis=0)

    view      = TIME_STEPS_PER_WEEK * 2
    train_idx = np.arange(len(train_raw))
    test_idx  = np.arange(len(train_raw), len(train_raw) + len(test_raw))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train_idx[-view:], train_raw[-view:],
            color="steelblue", alpha=0.4, linewidth=0.8, label="Train (last 2 weeks)")
    ax.plot(test_idx, test_raw,
            color="black", linewidth=0.9, label="Actual (test)")
    ax.plot(test_idx, median,
            color="red", linestyle="--", linewidth=1.2, label="Gaussian MC median")
    ax.fill_between(test_idx, lower, upper,
                    color="red", alpha=0.2,
                    label=f"{int((1 - alpha) * 100)}% Gaussian CI")
    ax.set_title(
        f"Monte Carlo forecast using ARMA({AR},{MA})\n"
        f"{N_SIMS} simulations, {int((1 - alpha) * 100)}% Confidence Interval"
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Traffic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return lower, upper, median


def print_coverage(test_raw: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> None:
    inside   = (test_raw >= lower) & (test_raw <= upper)
    coverage = inside.mean()
    widths   = upper - lower
    print(f"\n── Gaussian MC CI Summary ────────────────────────────────────")
    print(f"  Nominal coverage  : {(1 - alpha):.0%}")
    print(f"  Empirical coverage: {coverage:.1%}")
    print(f"  Mean CI width     : {widths.mean():.4f}")
    print(f"  Median CI width   : {np.median(widths):.4f}")
    print(f"──────────────────────────────────────────────────────────────\n")


def main():
    print("Loading data...")
    Tp = load_dataset()
    train_raw, test_raw = make_train_test_split(Tp, TRAIN_RATIO)
    steps = len(test_raw)

    print("Fitting ARMA model on doubly-differenced data...")
    results, seasonal_diff = fit_model(train_raw, AR, MA)
    resid = results.resid[~np.isnan(results.resid)]
    print(f"  AR params : {results.arparams}")
    print(f"  MA params : {results.maparams}")
    print(f"  σ̂         : {resid.std():.4f}  (used as Gaussian std)")

    print(f"\nRunning {N_SIMS} Gaussian MC simulations ({steps} steps ahead)...")
    rng   = np.random.default_rng(SEED)
    paths = gaussian_forecast(results, seasonal_diff, train_raw, steps, N_SIMS, rng)

    lower, upper, median = plot_results(train_raw, test_raw, paths, ALPHA)
    print_coverage(test_raw, lower, upper, ALPHA)


if __name__ == "__main__":
    main()
