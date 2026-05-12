from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy as sp
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
    #results = model.fit(method="innovations_mle") # Emils förslag
    return results, model


def check_stationarity(data: npt.NDArray) -> None:
    """
    Checks if the dataset is statiionary using the ADF test
    """
    result = adfuller(data, maxlag=TIME_STEPS_PER_DAY)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")



def forecast_coeff(n, results, max_h):

    p, q = len(results.arparams), len(results.maparams)
    sigma2 = results.params[p+q+1]
    
    roots = np.roots(np.append(-np.flip(results.arparams), 1)).astype(complex)
    #roots = np.roots([-results.arparams[3], -results.arparams[2],-results.arparams[1], -results.arparams[0], 1])
    # phi(t) = (t-r1)(t-r2)(t-r3)(t-r4) = r1r2r3r4(1-t/r1)(1-t/r2)(1-t/r3)(1-t/r4)

    psi = np.append(np.flip(results.maparams), 1).astype(complex)
    #psi = np.array([results.maparams[1], results.maparams[0], 1]).astype(complex)
    big_nr = 10000 # higher = more accuracy
    for i in range(len(roots)):
        r = roots[i]
        conv = np.zeros(big_nr).astype(complex)
        conv[0] = 1
        for j in range(1,big_nr):
            conv[j] = conv[j-1]*(1/r)
        psi = np.convolve(psi, conv)
    
    psi = np.real(psi)
    acf = np.zeros(n+max_h)

    for h in range(len(acf)):
    #for h in range(100):
        acf[h] = np.dot(psi[0:len(psi)-h], psi[h:len(psi)])

    acf = sigma2*acf

    predict_vectors = np.zeros((max_h, n)) # each row is a for a given h
    for h in range(1,max_h+1):
        predict_vectors[h-1, :] = sp.linalg.solve_toeplitz(acf[:n], acf[h:h+n])

    for k in [1,10,100,1000,n]:
        #k = 1 # n
        print(((acf[0] - np.dot(predict_vectors[0,:k], acf[1:k+1])) - sigma2)) 

    return predict_vectors


def forecast_perform(train_processed, predict_vectors, max_h):

    X_hat = np.zeros(max_h)

    for h in range(1,max_h+1):
        predict_vec = predict_vectors[h-1, :]
        a = np.flip(predict_vec)
        X_hat[h-1] = np.dot(train_processed, a)

    return X_hat


def main():
    np.random.seed(40)
    seasonal_lag = TIME_STEPS_PER_DAY*7
    nr_prediction_steps = TIME_STEPS_PER_DAY
    nr_of_training_weeks = 4
    training_length = TIME_STEPS_PER_DAY*7*nr_of_training_weeks
    data_length = TIME_STEPS_PER_DAY*7*(nr_of_training_weeks+1)
    Tp = load_dataset(length=data_length, start=27000)
    #train_raw, test_raw = make_train_test_split(Tp, training_length)
    train_raw = Tp.copy()[:training_length]
    train_processed = preprocess(train_raw, seasonal_lag)
    #ar = 1
    #ma = 2
    ar, ma = 4, 2
    mean_processed = np.mean(train_processed)
    empirical_variance_processed = np.mean((train_processed-mean_processed)**2)

    print("Mean of processed data:", mean_processed)
    print("Empirical variance of processed data:", empirical_variance_processed)
    results, _ = fit_model(train_processed, ar, ma)
    const = results.params[0] # Gives constant term in the model
    print(results.summary())
    #print(results.forecast(20))
    #correct_estimates = results.forecast(nr_prediction_steps)
    #print("påbörjar")
  
    predict_vectors = forecast_coeff(len(train_processed), results,nr_prediction_steps)

    n = TIME_STEPS_PER_DAY*6 + 1
    mse_ARMA = np.zeros(nr_prediction_steps)
    mse_noARMA = np.zeros(nr_prediction_steps)

    for shift in range(n): # shift = t
        if shift % 100 == 99 or shift == 0:
            print("Starting prediction", shift+1)
        #shift = np.random.randint(TIME_STEPS_PER_DAY*7)
        Tp_train_and_predict = Tp.copy()[shift:shift+training_length+nr_prediction_steps]
        train_raw, test_raw = make_train_test_split(Tp_train_and_predict, training_length)
        train_processed = preprocess(train_raw, seasonal_lag)
        prediction = forecast_perform(train_processed, predict_vectors, nr_prediction_steps)
        subtract_const = True
        if subtract_const:
            prediction = forecast_perform(train_processed-const, predict_vectors, nr_prediction_steps)+const
        prediction_arma = undo_preprocess(prediction, train_raw, seasonal_lag)
        prediction_noarma = undo_preprocess(nr_prediction_steps*[0], train_raw, seasonal_lag)
        #plt.plot(prediction_arma)
        #plt.plot(prediction_noarma)
        #plt.plot(test_raw)
        #plt.show()
        mse_ARMA += (prediction_arma-test_raw)**2
        mse_noARMA += (prediction_noarma-test_raw)**2

    mse_ARMA = mse_ARMA / n
    mse_noARMA = mse_noARMA / n
    plt.plot(np.arange(len(mse_ARMA))+1, mse_ARMA, label="ARMA")
    plt.plot(np.arange(len(mse_noARMA))+1, mse_noARMA, label="No ARMA")
    plt.title("ARMA vs No ARMA")
    plt.xlabel("Prediction Time Steps")
    plt.ylabel("MSE")
    plt.ylim(0,3.5)
    plt.legend(loc='upper left')
    plt.show()
        
    

    #plt.plot(undo_preprocess(estimates,train_raw,seasonal_lag),label="Linear")
    #plt.plot(undo_preprocess(len(estimates)*[0],train_raw,seasonal_lag),label="No-ARMA")
    #plt.plot(undo_preprocess(correct_estimates,train_raw,seasonal_lag),label="Built-in")
    #plt.plot(test_raw[:nr_prediction_steps],label="real")
    #plt.legend()
    #print(estimates-correct_estimates)
    
    #plt.plot(train_processed[-9:],label="Training")

    #print(train_processed[-9:])
    #plt.plot(estimates,label="Linear")
    #plt.plot(len(estimates)*[0],label="No-ARMA")
    #plt.plot(correct_estimates,label="Built-in")
    
    #plt.legend()
    #print(estimates-correct_estimates)
    
    #plt.show()

if __name__ == "__main__":
    main()