from itertools import product
from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller

TIME_STEPS_PER_DAY = 12 * 24
NODES_IN_DATASET = 12**2


def load_dataset(start: int = 0, length: int = 2000) -> npt.NDArray:
    """
    Returns the average trafic in the dataset.
    """
    T = np.load("./raw_data/abiline_ten.npy")
    Tp = np.sum(T, axis=(0, 1))[start : start + length]
    return Tp / NODES_IN_DATASET


def make_train_test_split(
    data: npt.NDArray, train_ratio: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    cutoff = int(len(data) * train_ratio)
    train = data[:cutoff]
    test = data[cutoff:]
    return train, test


def make_acf_plots(data: npt.NDArray) -> None:
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data, ax=ax1, lags=40)
    ax1.set_title("Autocorrelation Function (ACF)")
    plot_pacf(data, ax=ax2, lags=40, method="ywm")
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    plt.show()


def make_basic_plot(data: npt.NDArray) -> None:
    plt.plot(data)
    plt.grid(True, alpha=0.3)
    plt.show()


def preprocess(data: npt.NDArray) -> npt.NDArray:
    """
    Preprocessing function. Aim is to make the series stationary.
    """
    T = data.copy()
    T = np.diff(T, 1)
    seasonal_lag = TIME_STEPS_PER_DAY
    T = T[seasonal_lag:] - T[:-seasonal_lag]
    return T


def preprocess_test(test_raw: npt.NDArray, train_raw: npt.NDArray) -> npt.NDArray:
    """
    Applies the same differencing logic to the test set using training
    boundary values to avoid losing the first day of test data.
    """
    # Combine a bit of the end of train with test to calculate differences correctly
    lookback = TIME_STEPS_PER_DAY + 1
    combined = np.concatenate([train_raw[-lookback:], test_raw])

    # Apply the exact same steps as preprocess()
    diff1 = np.diff(combined, n=1)
    seasonal_diff = diff1[TIME_STEPS_PER_DAY:] - diff1[:-TIME_STEPS_PER_DAY]

    return seasonal_diff


def undo_preprocess(forecast_diff, train_raw, seasonal_lag):
    train_diff_1 = np.diff(train_raw, n=1)
    undone_diff_1 = np.zeros(len(forecast_diff))
    for i in range(len(forecast_diff)):
        val_lag = (
            train_diff_1[-(seasonal_lag - i)]
            if (seasonal_lag - i) > 0
            else undone_diff_1[i - seasonal_lag]
        )
        undone_diff_1[i] = forecast_diff[i] + val_lag
    forecast_final = np.cumsum(undone_diff_1) + train_raw[-1]
    return forecast_final


def fit_model(
    Tp: npt.NDArray, ar_component: int, ma_component: int
) -> Tuple[ARIMAResults, ARIMA]:
    model = ARIMA(Tp, order=(ar_component, 0, ma_component), enforce_stationarity=True)
    results = model.fit()
    return results, model


def check_stationarity(data: npt.NDArray) -> None:
    result = adfuller(data, maxlag=TIME_STEPS_PER_DAY)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")


def gridsearch(data, p_max, q_max):
    values = np.full((p_max + 1, q_max + 1), np.inf)
    for p, q in product(range(p_max + 1), range(q_max + 1)):
        try:
            model = ARIMA(data, order=(p, 0, q), enforce_stationarity=True)
            results = model.fit()

            if results.mle_retvals["converged"]:
                values[p, q] = results.bic
            else:
                values[p, q] = np.inf
        except Exception:
            continue

    if np.all(values == np.inf):
        print("All models failed to converge. Check your data stationarity!")
    else:
        print(values)
        best_p, best_q = np.unravel_index(np.argmin(values), values.shape)
        print(
            f"Best Valid Model: p={best_p}, q={best_q} | BIC={values[best_p, best_q]:.2f}"
        )


def plot_validation(
    train_data, test_data, forecast, title="Model Validation", ylabel="Value"
):
    """
    AI generated function
    """
    plt.figure(figsize=(12, 5))

    tail_len = min(len(train_data), TIME_STEPS_PER_DAY * 2)
    train_idx = np.arange(tail_len)
    plt.plot(
        train_idx,
        train_data[-tail_len:],
        label="Train (Recent)",
        color="blue",
        alpha=0.4,
    )

    test_idx = np.arange(tail_len, tail_len + len(test_data))
    plt.plot(test_idx, test_data, label="Actual (Test)", color="black", alpha=0.8)
    plt.plot(
        test_idx, forecast, label="Forecast", color="red", linestyle="--", linewidth=1.5
    )

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    n_days = 4
    start = TIME_STEPS_PER_DAY * n_days

    Tp = load_dataset(start=start, length=2000)
    train_raw, test_raw = make_train_test_split(Tp, 0.9)

    train_processed = preprocess(train_raw)
    gridsearch(train_processed, 10, 3)
    # check_stationarity(train_processed)
    # make_acf_plots(train_processed)
    exit()

    test_processed = preprocess_test(test_raw, train_raw)

    ar, ma = 4, 3
    results, model = fit_model(train_processed, ar, ma)
    print(results.summary())
    forecast_diff = results.forecast(steps=len(test_processed))
    forecast_final = undo_preprocess(forecast_diff, train_raw, TIME_STEPS_PER_DAY)

    plot_validation(
        train_raw,
        test_raw,
        forecast_final,
        title="Original Scale",
        ylabel="Traffic",
    )


if __name__ == "__main__":
    main()
