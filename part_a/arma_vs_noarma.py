from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller

# The dataset is is the
TIME_STEPS_PER_DAY = 12 * 24
NODES_IN_DATASET = 12**2


def load_dataset(length: int = 10080, start: int = 27000) -> npt.NDArray:
    """
    Returns the median traffic in the dataset across all node pairs.
    """
    T = np.load("./raw_data/abiline_ten.npy")
    Tp = np.median(T, axis=(0, 1))[start:start+length]
    return Tp


def make_train_test_split(
    data: npt.NDArray, cutoff: int
) -> Tuple[npt.NDArray, npt.NDArray]:
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


def preprocess(data: npt.NDArray, seasonal_lag: int) -> npt.NDArray:
    """
    Preprocessing function. Aim is to make the series sationary.
    """
    T = data.copy()
    T = np.diff(T, 1)
    #seasonal_lag = TIME_STEPS_PER_DAY*7
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


# Just nu är funkar modellen men ett problem som uppstår är att test datan hamnar på en helg
# så en modell anpassas på vecko data men utvärderas på helg data då trafiken är lägre.
# En lösning är bara att shifta vart vi börjar mäta.


def main():
    np.random.seed(40)
    seasonal_lag = TIME_STEPS_PER_DAY*7
    nr_prediction_steps = TIME_STEPS_PER_DAY
    nr_of_training_weeks = 4
    training_length = TIME_STEPS_PER_DAY*7*nr_of_training_weeks
    Tp = load_dataset(length=training_length + nr_prediction_steps)

    ar, ma = 1,2
    #ar, ma = 9,1
    n = 100
    mse_ARMA = np.zeros(nr_prediction_steps)
    mse_noARMA = np.zeros(nr_prediction_steps)

    for i in range(n):
        print("Starting prediction", i+1)
        prediction_time = np.random.randint(TIME_STEPS_PER_DAY*7) # random time point during fourth week
        #last_training_step = TIME_STEPS_PER_DAY*7*(nr_of_training_weeks-1) + prediction_time
        last_training_step = TIME_STEPS_PER_DAY*7*(nr_of_training_weeks-1) + prediction_time
        Tp_copy = Tp.copy()
        #Tp_train_and_predict = Tp_copy[0:last_training_step + nr_prediction_steps]
        Tp_train_and_predict = Tp_copy[prediction_time:last_training_step + nr_prediction_steps]
        train_raw, test_raw = make_train_test_split(Tp_train_and_predict, last_training_step-prediction_time)
        train_processed = preprocess(train_raw, seasonal_lag)

        results, _ = fit_model(train_processed, ar, ma)
        forecast_diff = results.forecast(steps=nr_prediction_steps)
        forecast_final = undo_preprocess(forecast_diff, train_raw, seasonal_lag) # With ARMA
        forecast_final0 = undo_preprocess([0]*len(forecast_diff), train_raw, seasonal_lag) # With ARMA put to 0
        mse_ARMA += (forecast_final-test_raw)**2
        mse_noARMA += (forecast_final0-test_raw)**2
    mse_ARMA = mse_ARMA / n
    mse_noARMA = mse_noARMA / n
    plt.plot(np.arange(len(mse_ARMA))+1, mse_ARMA , label="ARMA")
    plt.plot(np.arange(len(mse_noARMA))+1, mse_noARMA, label="No ARMA")
    plt.title("ARMA vs No ARMA")
    plt.xlabel("Prediction Time Steps")
    plt.ylabel("MSE")
    plt.legend(loc='upper left')
    plt.show()
    #check_stationarity(train_processed)
    #make_basic_plot(train_processed)
    #make_acf_plots(train_processed)
    #exit()



if __name__ == "__main__":
    main()
