from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tsa.stattools import adfuller

# Constants from the original source
TIME_STEPS_PER_DAY = 12 * 24
TIME_STEPS_PER_WEEK = 12 * 24 * 7
NODES_IN_DATASET = 12**2

def load_dataset(length: int = 10080, start: int = 27000) -> npt.NDArray:
    """Returns the median traffic in the dataset across all node pairs."""
    # Note: Ensure the file path matches your local environment
    T = np.load("./raw_data/abiline_ten.npy")
    Tp = np.median(T, axis=(0, 1))[start:start+length]
    return Tp

def make_train_test_split(
    data: npt.NDArray, train_ratio: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Splits data into training and testing sets[cite: 1]."""
    cutoff = int(len(data) * train_ratio)
    train = data[:cutoff]
    test = data[cutoff:]
    return train, test

def fit_integrated_model(
    train_data: npt.NDArray, ar: int, ma: int
) -> SARIMAXResults:
    """
    Fits a SARIMAX model that handles differencing natively.
    - order (ar, 1, ma): The '1' replaces np.diff(data, 1)[cite: 1].
    - seasonal_order (0, 1, 0, lag): The middle '1' replaces the seasonal subtraction[cite: 1].
    """
    model = SARIMAX(
        train_data,
        order=(ar, 1, ma),
        seasonal_order=(0, 1, 0, TIME_STEPS_PER_WEEK),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def plot_validation(train_raw, test_raw, forecast, lower_bound=None, upper_bound=None, 
                    title="Model Validation"):
    """Visualizes the forecast against the actual test data[cite: 1]."""
    plt.figure(figsize=(12, 6))

    # Plot training data (limited to last few weeks for clarity)
    view_limit = TIME_STEPS_PER_WEEK * 2
    train_idx = np.arange(len(train_raw))
    plt.plot(train_idx[-view_limit:], train_raw[-view_limit:], label="Train (Recent)", color="blue", alpha=0.3)

    # Plot Test vs Forecast
    test_idx = np.arange(len(train_raw), len(train_raw) + len(test_raw))
    plt.plot(test_idx, test_raw, label="Actual (Test)", color="black", alpha=0.7)
    plt.plot(test_idx, forecast, label="Forecast", color="red", linestyle="--")

    # Add Shaded Prediction Interval
    if lower_bound is not None and upper_bound is not None:
        plt.fill_between(test_idx, lower_bound, upper_bound, color='red', alpha=0.2, label="95% Prediction Interval")

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # 1. Load and Split Raw Data[cite: 1]
    Tp = load_dataset()
    train_raw, test_raw = make_train_test_split(Tp, 0.8)
    steps = len(test_raw)

    # 2. Fit the Integrated Model
    # We use (1, 1, 2) to match your original differencing logic[cite: 1]
    ar, ma = 1, 2
    print("Fitting SARIMAX model (this may take a moment due to seasonal integration)...")
    results = fit_integrated_model(train_raw, ar, ma)

    # 3. Get Forecast and Confidence Intervals
    # statsmodels now calculates the expansion math automatically
    forecast_obj = results.get_forecast(steps=steps)
    forecast_final = forecast_obj.predicted_mean
    
    # Retrieve the confidence intervals (95% by default)
    bounds = forecast_obj.conf_int(alpha=0.05)
    # If bounds is a DataFrame, use iloc; if it's a numpy array, use slicing
    if hasattr(bounds, 'iloc'):
        lower_final = bounds.iloc[:, 0]
        upper_final = bounds.iloc[:, 1]
    else:
        lower_final = bounds[:, 0]
        upper_final = bounds[:, 1]

    # 4. Plot Results in the Original Domain
    plot_validation(
        train_raw, 
        test_raw, 
        forecast_final, 
        lower_bound=lower_final, 
        upper_bound=upper_final, 
        title="Integrated SARIMAX: Native Original Domain Forecast"
    )

    print(results.summary())

if __name__ == "__main__":
    main()