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
    return results, model


def check_stationarity(data: npt.NDArray) -> None:
    """
    Checks if the dataset is statiionary using the ADF test
    """
    result = adfuller(data, maxlag=TIME_STEPS_PER_DAY)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")



def forecast_coeff_old(n, results, max_h, compute_only_one_step_predictors):  

    sigma2 = results.params[4]
    phi = results.arparams[0]
    theta1,theta2 = results.maparams[0],results.maparams[1]


    # Computing acf for ARMA(1,2) process
    acf_jesper = np.zeros(n + max_h) # +1???
    acf_jesper[0] = sigma2*(1+(phi+theta1)**2 + (phi**2 + theta1*phi + theta2)**2 / (1 - phi**2))
    acf_jesper[1] = sigma2*((phi+theta1)*(1 + phi**2 + theta1*phi + theta2) + (phi**2 + theta1*phi + theta2)**2 *phi / (1 - phi**2))
    acf_jesper[2] = sigma2*(phi**2 + theta1*phi + theta2)*(1 + (phi + theta1)*phi + (phi**2 + theta1*phi + theta2)*phi**2 / (1 - phi**2))
    for i in range(3,len(acf_jesper)):
        acf_jesper[i] = acf_jesper[i-1]*phi

    acf = acf_jesper

    if compute_only_one_step_predictors:

        predict_vectors = sp.linalg.solve_toeplitz(acf[:n], acf[1:1+n])

        return predict_vectors
    
    else:

        predict_vectors = np.zeros((max_h, n)) # each row is a for a given h
        for h in range(1,max_h+1):
            predict_vectors[h-1, :] = sp.linalg.solve_toeplitz(acf[:n], acf[h:h+n])

        return predict_vectors
    
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

    return predict_vectors


def forecast_perform(train_processed, predict_vectors, max_h):

    X_hat = np.zeros(max_h)

    for h in range(1,max_h+1):
        predict_vec = predict_vectors[h-1, :]
        a = np.flip(predict_vec)
        X_hat[h-1] = np.dot(train_processed, a)

    return X_hat


def stample_stationary(train_processed, predict_vector, max_h, sigma2):
    
    a = np.flip(predict_vector)
    n = len(train_processed)
    train_and_sampe = np.zeros(n+max_h)
    train_and_sampe[:len(train_processed)] = train_processed

    for h in range(max_h):

        Zt = np.random.normal(0,1)*sigma2**0.5 # We assume normally distributed WN
        #print(Zt)
        train_and_sampe[n+h] = np.dot(train_and_sampe[h:n+h], a) + Zt

    sample = train_and_sampe[n:]

    return sample

def main():
    np.random.seed(40)
    seasonal_lag = TIME_STEPS_PER_DAY*7
    nr_prediction_steps = TIME_STEPS_PER_DAY
    #nr_prediction_steps = TIME_STEPS_PER_DAY*7
    nr_of_training_weeks = 4 # 4
    training_length = TIME_STEPS_PER_DAY*7*nr_of_training_weeks
    data_length = TIME_STEPS_PER_DAY*7*(nr_of_training_weeks+1)
    Tp = load_dataset(length=data_length + nr_prediction_steps,start=27000)
    #train_raw, test_raw = make_train_test_split(Tp, training_length)
    train_raw = Tp.copy()[:training_length]
    train_processed = preprocess(train_raw, seasonal_lag)
    #ar = 1
    #ma = 2
    ar, ma = 4, 2
    results, _ = fit_model(train_processed, ar, ma)
    print(results.summary())
    const = results.params[0] # Gives constant term in the model
    sigma2 = results.params[ar+ma+1]
    #sigma2 = results.params[4]
    #print(results.forecast(20))
    #correct_estimates = results.forecast(nr_prediction_steps)
    #print("påbörjar")
    #predict_vector = forecast_coeff(len(train_processed),results,nr_prediction_steps,True)
    predict_vectors = forecast_coeff(len(train_processed),results,nr_prediction_steps)
    predict_vector = predict_vectors[0,:]
    nr_samples = 10000
    #mse_ARMA = np.zeros(nr_prediction_steps)
    #mse_noARMA = np.zeros(nr_prediction_steps)
    sampled_forecasts = np.zeros((nr_samples, nr_prediction_steps)) # randomized forecasts in stationary domains translated to time domain
    Tp_train_and_predict = Tp.copy()[0:training_length+nr_prediction_steps]
    train_raw, test_raw = make_train_test_split(Tp_train_and_predict, training_length)

    vi_struntar_i_konstanten = False
    if vi_struntar_i_konstanten:
        const = 0

    train_processed = preprocess(train_raw, seasonal_lag)-const
   

    for i in range(nr_samples):
        if i % 1000 == 999 or i == 0:
            print("Starting prediction", i+1)
        prediction = stample_stationary(train_processed, predict_vector, nr_prediction_steps, sigma2)+const
        prediction_arma = undo_preprocess(prediction, train_raw, seasonal_lag)
        #prediction_noarma = undo_preprocess(nr_prediction_steps*[0], train_raw, seasonal_lag)
        sampled_forecasts[i, :] = prediction_arma
        #plt.plot(prediction_arma)
        #plt.plot(prediction_arma, label = "arma")
        #plt.plot(prediction_noarma, label = "noarma")
        #plt.plot(test_raw, label = "test raw")
        #plt.legend()
        #plt.show()
    #plt.show()

    alpha = 0.05 # degree of confidence

    lower_bounds = np.zeros(nr_prediction_steps)
    upper_bounds = np.zeros(nr_prediction_steps)
    optimal_stationary_forecast = forecast_perform(train_processed, predict_vectors, nr_prediction_steps)+const
    #optimal_stationary_forecast = stample_stationary(train_processed, predict_vector, nr_prediction_steps, sigma2=0)+const
    optimal_forecast = undo_preprocess(optimal_stationary_forecast, train_raw, seasonal_lag)

    for i in range(nr_prediction_steps):
        samples = sampled_forecasts[:, i]
        samples.sort()
        lower_bounds[i] = (samples[int(nr_samples*alpha/2)-1] + samples[int(nr_samples*alpha/2)])/2
        upper_bounds[i] = (samples[nr_samples - int(nr_samples*alpha/2)-1] + samples[nr_samples - int(nr_samples*alpha/2)])/2

    plt.plot(np.arange(nr_prediction_steps)+1, lower_bounds, 'r', linestyle = 'dashed', label = str(int(100*(1-alpha))) + "% confidence bounds")
    plt.plot(np.arange(nr_prediction_steps)+1, upper_bounds, 'r', linestyle = 'dashed')
    plt.plot(np.arange(nr_prediction_steps)+1, optimal_forecast, label = "Linear prediction")
    plt.plot()
    plt.plot(np.arange(nr_prediction_steps)+1, test_raw, label = 'Validation data')
    plt.legend()
    plt.show()    


        
    
if __name__ == "__main__":
    main()