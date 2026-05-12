"""
Bootstrap confidence intervals for future predictions via Monte Carlo resampling.

The model is ARIMA(AR, 1, MA) fitted on the seasonally-differenced series
    d_t = X_t - X_{t-2016}

To simulate H steps ahead without assuming Gaussian innovations:
  1. Resample residuals ε* with replacement from the fitted residuals.
  2. Propagate the ARMA recursion forward using ε* to get simulated d̂ paths.
  3. Reconstruct the original domain:
         X̂_{T+h} = d̂_{T+h} + X_{T+h-2016}
     The lag-2016 values are always in the training set when steps < 2016.
  4. Collect N_SIMS paths and take empirical quantiles for the CI.
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from main_anton import load_dataset, make_train_test_split, TIME_STEPS_PER_WEEK

# ── Configuration ────────────────────────────────────────────────────────────
AR          = 4
MA          = 2
N_SIMS      = 1000
ALPHA       = 0.05     # 1 - ALPHA confidence interval
TRAIN_RATIO = 0.9
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────


def fit_model(train_data: np.ndarray, ar: int, ma: int):
    """
    ARIMA(ar, 0, ma) on the seasonally-differenced series.
    Both diffs are applied manually: seasonal (period=2016) then regular.
    The model sees only the stationary W_t series with no internal differencing.
    """
    seasonal_diff = train_data[TIME_STEPS_PER_WEEK:] - train_data[:-TIME_STEPS_PER_WEEK]
    W = np.diff(seasonal_diff)
    model = SARIMAX(
        W,
        order=(ar, 0, ma),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)
    return results, seasonal_diff


def bootstrap_forecast(
    results,
    seasonal_diff: np.ndarray,
    train_data: np.ndarray,
    steps: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns an (n_sims, steps) array of bootstrap forecast paths
    in the original (un-differenced) domain.

    The ARMA(p, q) recursion on W_t = d_t - d_{t-1}:
        W_{T+h} = Σ φ_i * W_{T+h-i}  +  ε*_{T+h}  +  Σ θ_j * ε_{T+h-j}
    Then d_{T+h} = d_T + Σ_{k=1}^{h} W_{T+k}
    Then X_{T+h} = d_{T+h} + X_{T+h-2016}
    """
    ar_params = results.arparams          # shape (p,)
    ma_params = results.maparams          # shape (q,)
    p, q      = len(ar_params), len(ma_params)

    residuals = np.asarray(results.resid)

    # Regular differences of the seasonal-diff series (seed for recursion)
    W = np.diff(seasonal_diff)

    W_seed   = W[-(max(p, 1)):]
    eps_seed = residuals[-(max(q, 1)):]
    last_d   = seasonal_diff[-1]

    # X_{T+h-2016} for h = 1..steps — all within the training set
    T             = len(train_data)
    seasonal_base = train_data[T - TIME_STEPS_PER_WEEK : T - TIME_STEPS_PER_WEEK + steps]

    paths = np.empty((n_sims, steps))

    for s in range(n_sims):
        eps_future = rng.choice(residuals, size=steps, replace=True)

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
            color="red", linestyle="--", linewidth=1.2, label="Bootstrap median")
    ax.fill_between(test_idx, lower, upper,
                    color="red", alpha=0.2,
                    label=f"{int((1 - alpha) * 100)}% bootstrap CI")
    ax.set_title(
        f"Monte Carlo bootstrap forecast — ARIMA({AR},0,{MA}) on seasonally differenced data\n"
        f"{N_SIMS} simulations, {int((1 - alpha) * 100)}% CI"
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
    print(f"\n── Bootstrap CI Summary ──────────────────────────────────────")
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

    print("Fitting ARIMA model on seasonally differenced data...")
    results, seasonal_diff = fit_model(train_raw, AR, MA)
    resid = results.resid[~np.isnan(results.resid)]
    print(f"  AR params : {results.arparams}")
    print(f"  MA params : {results.maparams}")
    print(f"  σ̂         : {resid.std():.4f}")

    print(f"\nRunning {N_SIMS} Monte Carlo bootstrap simulations ({steps} steps ahead)...")
    rng   = np.random.default_rng(SEED)
    paths = bootstrap_forecast(results, seasonal_diff, train_raw, steps, N_SIMS, rng)

    lower, upper, median = plot_results(train_raw, test_raw, paths, ALPHA)
    print_coverage(test_raw, lower, upper, ALPHA)


if __name__ == "__main__":
    main()
