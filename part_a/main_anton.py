from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller

# The dataset is is the
TIME_STEPS_PER_DAY = 12 * 24
TIME_STEPS_PER_WEEK = 12 * 24 * 7
NODES_IN_DATASET = 12**2

"""
def load_dataset(length: int = 6000, start: int = 20000) -> npt.NDArray:
    T = np.load("./raw_data/abiline_ten.npy")
    Tp = np.sum(np.sum(T, axis=0), axis=0)[start:start+length]
    return Tp / NODES_IN_DATASET
"""

#"""
def load_dataset(length: int = 6000, start: int = 26070) -> npt.NDArray:
    """
    Returns the median traffic in the dataset across all node pairs.
    """
    T = np.load("./raw_data/abiline_ten.npy")
    Tp = np.median(T, axis=(0, 1))[start:start+length]
    return Tp
#"""


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
    Preprocessing function. Aim is to make the series sationary.
    """
    T = data.copy()
    T = np.diff(T, 1)
    seasonal_lag = TIME_STEPS_PER_WEEK
    T = T[seasonal_lag:] - T[:-seasonal_lag]
    return T


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
    """
    Fits model...
    """
    model = ARIMA(Tp, order=(ar_component, 0, ma_component))
    results = model.fit()
    return results, model


def check_stationarity(data: npt.NDArray) -> None:
    """
    Checks if the dataset is statiionary using the ADF test
    """
    result = adfuller(data, maxlag=TIME_STEPS_PER_DAY)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")


def plot_stationary_forecast(train_processed: npt.NDArray, forecast_diff: npt.NDArray, 
    title: str = "Stationary Series and Forecast (Differenced Domain)"
) -> None:
    """
    Plots the processed (stationary) training data alongside the forecast
    in the differenced domain.

    AI generated
    """
    plt.figure(figsize=(12, 6))

    # Plot the stationary training data
    # Note: We plot the last portion to avoid an overly dense graph if the series is long
    display_cutoff = max(0, len(train_processed) - 1000) 
    train_idx = range(display_cutoff, len(train_processed))
    
    plt.plot(train_idx, train_processed[display_cutoff:], label="Processed Train (Stationary)", color="blue", alpha=0.5)

    # Plot the forecast in the stationary domain
    forecast_idx = np.arange(len(train_processed), len(train_processed) + len(forecast_diff))
    plt.plot(forecast_idx, forecast_diff, label="Forecast (Differenced)", color="red", linestyle="--")

    # Add a horizontal line at zero to represent the expected mean of the stationary series
    plt.axhline(0, color='black', linestyle=':', alpha=0.5, label='Zero Mean')

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Differenced Traffic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_validation(train_raw, test_raw, forecast, lower_bound=None, upper_bound=None, 
                    title="Model Validation"):
    """
    AI generated function
    """
    plt.figure(figsize=(12, 6))

    # Plot training data (optional, can be messy if too long)
    plt.plot(range(len(train_raw)), train_raw, label="Train", color="blue", alpha=0.3)

    # Plot Test vs Forecast
    test_idx = np.arange(len(train_raw), len(train_raw) + len(test_raw))
    plt.plot(test_idx, test_raw, label="Actual (Test)", color="black", alpha=0.7)
    plt.plot(test_idx, forecast, label="Forecast", color="red", linestyle="--")

    # Add Shaded Prediction Interval
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(test_idx, lower_bound, upper_bound, color='red', alpha=0.2, label="95% Prediction Interval")

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Normalized Traffic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Just nu är funkar modellen men ett problem som uppstår är att test datan hamnar på en helg
# så en modell anpassas på vecko data men utvärderas på helg data då trafiken är lägre.
# En lösning är bara att shifta vart vi börjar mäta.


def main():
    Tp = load_dataset()
    train_raw, test_raw = make_train_test_split(Tp, 0.9)
    train_processed = preprocess(train_raw)
    #check_stationarity(train_processed)
    # make_basic_plot(train_processed)
    # make_acf_plots(train_processed)
    # exit()

    ar, ma = 1, 2
    results, _ = fit_model(train_processed, ar, ma)

    steps = len(test_raw)
    
    # --- ADDED (Gemini): Forecast extraction and interval calculation ---
    forecast_obj = results.get_forecast(steps=steps)
    forecast_diff = forecast_obj.predicted_mean
    se_diff = forecast_obj.se_mean
    
    forecast_final = undo_preprocess(forecast_diff, train_raw, TIME_STEPS_PER_WEEK)
    
    h_array = np.arange(1, steps + 1)
    se_integrated = se_diff * np.sqrt(h_array)
    
    lower_bound = forecast_final - (1.96 * se_integrated)
    upper_bound = forecast_final + (1.96 * se_integrated)
    # -----------------------------------------------------------

    print(results.summary())
    
    # UPDATED: Pass bounds to plotting function
    plot_validation(train_raw, test_raw, forecast_final, lower_bound=lower_bound, upper_bound=upper_bound)


if __name__ == "__main__":
    main()