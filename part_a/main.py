import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

T = np.load("./raw_data/abiline_ten.npy")
Tp = np.sum(np.sum(T, axis=0), axis=0)[:2000]
Tp = Tp / np.mean(Tp)

# plt.plot(Tp)
# plt.show()


def make_acf_plots(Tp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(Tp, ax=ax1, lags=40)
    ax1.set_title("Autocorrelation Function (ACF)")
    plot_pacf(Tp, ax=ax2, lags=40, method="ywm")
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    plt.tight_layout()
    plt.show()


Tp = np.diff(Tp, 1)
seasonal_lag = 12 * 24
Tp_final = Tp[seasonal_lag:] - Tp[:-seasonal_lag]
make_acf_plots(Tp_final)
