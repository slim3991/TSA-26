from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

T = np.load("./raw_data/abiline_ten.npy")
Tp = np.sum(np.sum(T, axis=0), axis=0)[:2000]
Tp = Tp / np.mean(Tp)

# plt.plot(Tp)
# plt.show()


def make_train_test_split(
    data: npt.NDArray, train_ratio: float
) -> Tuple[npt.NDArray, npt.NDArray]:
    T = data.copy()
    len = data.shape[0]
    cutoff = len // train_ratio
    train = T[:cutoff]
    test = T[cutoff:]

    return train, test


def make_acf_plots(data: npt.NDArray) -> None:
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data, ax=ax1, lags=40)
    ax1.set_title("Autocorrelation Function (ACF)")
    plot_pacf(data, ax=ax2, lags=40, method="ywm")
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    plt.show()


def preprocess(data: npt.NDArray) -> npt.NDArray:
    T = data.copy()
    data = np.diff(T, 1)
    seasonal_lag = 12 * 24
    T_final = T[seasonal_lag:] - T[:-seasonal_lag]
    return T_final


def undo_preprocess(Tp: npt.NDArray) -> npt.NDArray: ...  # TODO


def fit_model(Tp: npt.NDArray, ar_component: int, ma_component: int):
    model = ARIMA(Tp, order=(ar_component, 0, ma_component))
    results = model.fit()
    print(results.summary())


Tp_final = preprocess(Tp)
make_acf_plots(Tp_final)
