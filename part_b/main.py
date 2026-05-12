from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import warnings
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.sm_exceptions import ConvergenceWarning

DATA_DIR = Path(__file__).resolve().parent
PROPANE_CSV = DATA_DIR / "Propane.csv"
DATE_COLUMN = "observation_date"
PRICE_COLUMN = "DPROPANEMBTX"
PRICE_SOURCE = "FRED / U.S. Energy Information Administration"
PRICE_UNITS = "dollars per gallon"
TEST_SIZE = 20
MAX_ARMA_P = 5
MAX_ARMA_Q = 5
RUN_GRID_SEARCH = False
HARDCODED_ARMA_ORDER = (1, 0)
FORECAST_INTERVAL_ALPHA = 0.05
FORECAST_SIMULATIONS = 10_000
FORECAST_RANDOM_SEED = 12345


@dataclass(frozen=True)
class TrainTestSplit:
    train_return_dates: npt.NDArray[np.datetime64]
    train_returns: npt.NDArray[np.float64]
    test_return_dates: npt.NDArray[np.datetime64]
    test_returns: npt.NDArray[np.float64]
    train_price_dates: npt.NDArray[np.datetime64]
    train_prices: npt.NDArray[np.float64]
    test_price_dates: npt.NDArray[np.datetime64]
    test_prices: npt.NDArray[np.float64]
    last_train_date: np.datetime64
    last_train_price: float


@dataclass(frozen=True)
class ARMAGridResult:
    p: int
    q: int
    aic: float
    bic: float
    converged: bool
    error: str | None = None


@dataclass(frozen=True)
class ARMAGridSearch:
    results: list[ARMAGridResult]
    best_result: ARMAGridResult


@dataclass(frozen=True)
class ForecastResult:
    dates: npt.NDArray[np.datetime64]
    actual_prices: npt.NDArray[np.float64]
    arma_return_forecast: npt.NDArray[np.float64]
    arma_price_forecast: npt.NDArray[np.float64]
    arma_price_lower: npt.NDArray[np.float64]
    arma_price_upper: npt.NDArray[np.float64]


@dataclass(frozen=True)
class ForecastErrors:
    model_name: str
    mae: float
    rmse: float
    mape: float


def load_propane(
    file_path: Path = PROPANE_CSV,
) -> Tuple[npt.NDArray[np.datetime64], npt.NDArray[np.float64]]:
    raw = np.genfromtxt(
        file_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        missing_values="",
        filling_values=np.nan,
    )

    dates = raw[DATE_COLUMN].astype("datetime64[D]")
    prices = raw[PRICE_COLUMN].astype(np.float64)
    missing_prices = int(np.isnan(prices).sum())
    observed_prices = len(prices) - missing_prices

    print("Raw propane data")
    print(f"Source: {PRICE_SOURCE} ({PRICE_COLUMN})")
    print(f"Units: {PRICE_UNITS}")
    print(f"Rows in file: {len(prices)}")
    print(f"Observed prices: {observed_prices}")
    print(f"Missing prices: {missing_prices}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Min price: {np.nanmin(prices):.3f}")
    print(f"Max price: {np.nanmax(prices):.3f}")

    return dates, prices


def load_data(file_path: Path) -> npt.NDArray[np.float64]:
    _, prices = load_propane(file_path)
    return prices


def clean_data(
    dates: npt.NDArray[np.datetime64],
    prices: npt.NDArray[np.float64],
) -> Tuple[
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float64],
]:
    valid_price_mask = np.isfinite(prices) & (prices > 0)
    dropped_missing = int(np.isnan(prices).sum())
    dropped_nonpositive = int((np.isfinite(prices) & (prices <= 0)).sum())

    clean_dates = dates[valid_price_mask]
    clean_prices = prices[valid_price_mask]

    log_prices = np.log(clean_prices)
    log_returns = np.diff(log_prices)
    return_dates = clean_dates[1:]

    print("\nCleaned propane data")
    print(f"Dropped missing prices: {dropped_missing}")
    print(f"Dropped non-positive prices: {dropped_nonpositive}")
    print(f"Observed price rows used: {len(clean_prices)}")
    print(f"Log-return observations: {len(log_returns)}")

    return clean_dates, clean_prices, log_prices, return_dates, log_returns


def make_train_test_split(
    clean_dates: npt.NDArray[np.datetime64],
    clean_prices: npt.NDArray[np.float64],
    return_dates: npt.NDArray[np.datetime64],
    log_returns: npt.NDArray[np.float64],
    test_size: int = TEST_SIZE,
) -> TrainTestSplit:
    if test_size <= 0:
        raise ValueError("test_size must be positive.")
    if len(clean_prices) != len(log_returns) + 1:
        raise ValueError("Price series must be exactly one element longer than returns.")
    if len(clean_prices) <= test_size:
        raise ValueError("Not enough observations for the requested test split.")

    split_index = len(clean_prices) - test_size
    train_prices = clean_prices[:split_index]
    train_price_dates = clean_dates[:split_index]
    test_prices = clean_prices[split_index:]
    test_price_dates = clean_dates[split_index:]

    train_returns = log_returns[: split_index - 1]
    train_return_dates = return_dates[: split_index - 1]
    test_returns = log_returns[split_index - 1 :]
    test_return_dates = return_dates[split_index - 1 :]

    split = TrainTestSplit(
        train_return_dates=train_return_dates,
        train_returns=train_returns,
        test_return_dates=test_return_dates,
        test_returns=test_returns,
        train_price_dates=train_price_dates,
        train_prices=train_prices,
        test_price_dates=test_price_dates,
        test_prices=test_prices,
        last_train_date=train_price_dates[-1],
        last_train_price=float(train_prices[-1]),
    )

    print("\nTrain/test split")
    print(f"Training log returns: {len(split.train_returns)}")
    print(f"Test log returns: {len(split.test_returns)}")
    print(f"Training price observations: {len(split.train_prices)}")
    print(f"Test price observations: {len(split.test_prices)}")
    print(f"Last training price date: {split.last_train_date}")
    print(f"Last training price: {split.last_train_price:.3f}")
    print(f"Test price date range: {split.test_price_dates[0]} to {split.test_price_dates[-1]}")

    return split


def grid_search_arma(
    train_returns: npt.NDArray[np.float64],
    p_max: int = MAX_ARMA_P,
    q_max: int = MAX_ARMA_Q,
) -> ARMAGridSearch:
    if p_max < 0 or q_max < 0:
        raise ValueError("p_max and q_max must be non-negative.")

    finite_returns = train_returns[np.isfinite(train_returns)]
    if len(finite_returns) != len(train_returns):
        raise ValueError("Training returns contain missing or infinite values.")

    results = []

    print("\nARMA grid search on training log returns")
    print("Order      AIC          BIC          Converged")

    for p, q in product(range(p_max + 1), range(q_max + 1)):
        try:
            model = ARIMA(
                finite_returns,
                order=(p, 0, q),
                enforce_stationarity=True,
                enforce_invertibility=True,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Non-stationary starting autoregressive parameters found.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Non-invertible starting MA parameters found.*",
                    category=UserWarning,
                )
                fitted_model = model.fit()

            converged = bool(fitted_model.mle_retvals.get("converged", False))
            result = ARMAGridResult(
                p=p,
                q=q,
                aic=float(fitted_model.aic),
                bic=float(fitted_model.bic),
                converged=converged,
            )
            print(
                f"({p},0,{q})  {result.aic:10.2f}  {result.bic:10.2f}  "
                f"{result.converged}"
            )
        except Exception as exc:
            result = ARMAGridResult(
                p=p,
                q=q,
                aic=np.inf,
                bic=np.inf,
                converged=False,
                error=str(exc),
            )
            print(f"({p},0,{q})  {'failed':>10}  {'failed':>10}  False")

        results.append(result)

    valid_results = [
        result
        for result in results
        if result.converged and np.isfinite(result.bic)
    ]
    if not valid_results:
        raise RuntimeError("No ARMA models converged during grid search.")

    best_result = min(valid_results, key=lambda result: result.bic)
    print(
        "\nSelected model by BIC: "
        f"ARMA({best_result.p}, {best_result.q}) "
        f"with BIC={best_result.bic:.2f}"
    )

    return ARMAGridSearch(results=results, best_result=best_result)


def fit_final_arma_model(
    train_returns: npt.NDArray[np.float64],
    selected_model: ARMAGridResult,
    residual_lags: int = 40,
) -> ARIMAResults:
    finite_returns = train_returns[np.isfinite(train_returns)]
    if len(finite_returns) != len(train_returns):
        raise ValueError("Training returns contain missing or infinite values.")

    p, q = selected_model.p, selected_model.q
    model = ARIMA(
        finite_returns,
        order=(p, 0, q),
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message="Non-stationary starting autoregressive parameters found.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting MA parameters found.*",
            category=UserWarning,
        )
        fitted_model = model.fit()

    converged = bool(fitted_model.mle_retvals.get("converged", False))

    print(f"\nFinal ARMA({p}, {q}) model on training log returns")
    print(f"Converged: {converged}")
    print(f"AIC: {fitted_model.aic:.2f}")
    print(f"BIC: {fitted_model.bic:.2f}")
    print(f"Log likelihood: {fitted_model.llf:.2f}")

    print("\nParameter estimates")
    print("Parameter       Estimate      Std. error")
    for name, estimate, std_error in zip(
        fitted_model.param_names,
        fitted_model.params,
        fitted_model.bse,
    ):
        print(f"{name:<12}  {estimate:12.6f}  {std_error:12.6f}")

    residuals = np.asarray(fitted_model.resid, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]

    print("\nResidual summary")
    print(f"Observations: {len(residuals)}")
    print(f"Mean: {np.mean(residuals):.6f}")
    print(f"Standard deviation: {np.std(residuals, ddof=1):.6f}")
    print(f"Minimum: {np.min(residuals):.6f}")
    print(f"Maximum: {np.max(residuals):.6f}")

    lags = min(residual_lags, len(residuals) // 2 - 1)
    _, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals, ax=ax, lags=lags)
    ax.set_title(f"Residual ACF for ARMA({p}, {q})")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    plt.tight_layout()
    plt.show()

    return fitted_model


def forecast_prices(
    fitted_model: ARIMAResults,
    split: TrainTestSplit,
    alpha: float = FORECAST_INTERVAL_ALPHA,
    simulations: int = FORECAST_SIMULATIONS,
    random_seed: int = FORECAST_RANDOM_SEED,
) -> ForecastResult:
    horizon = len(split.test_prices)
    forecast_returns = np.asarray(fitted_model.forecast(steps=horizon), dtype=np.float64)

    last_train_log_price = np.log(split.last_train_price)
    forecast_log_prices = last_train_log_price + np.cumsum(forecast_returns)
    arma_price_forecast = np.exp(forecast_log_prices)

    simulated_returns = np.asarray(
        fitted_model.simulate(
            nsimulations=horizon,
            repetitions=simulations,
            anchor="end",
            random_state=random_seed,
        ),
        dtype=np.float64,
    )
    if simulated_returns.ndim == 3:
        simulated_returns = simulated_returns[:, 0, :]
    elif simulated_returns.ndim == 1:
        simulated_returns = simulated_returns[:, np.newaxis]

    simulated_log_prices = last_train_log_price + np.cumsum(simulated_returns, axis=0)
    simulated_prices = np.exp(simulated_log_prices)
    arma_price_lower = np.quantile(simulated_prices, alpha / 2, axis=1)
    arma_price_upper = np.quantile(simulated_prices, 1 - alpha / 2, axis=1)
    interval_coverage = np.mean(
        (split.test_prices >= arma_price_lower) & (split.test_prices <= arma_price_upper)
    )

    print("\nForecast")
    print(f"Forecast horizon: {horizon} observed trading days")
    print(
        "Method: ARMA conditional mean forecasts for log returns, "
        "accumulated from the last training price."
    )
    print(f"Prediction interval: {(1 - alpha) * 100:.0f}% simulated price interval")
    print(f"Last training price: {split.last_train_price:.3f}")
    print(f"First ARMA forecast price: {arma_price_forecast[0]:.3f}")
    print(f"Last ARMA forecast price: {arma_price_forecast[-1]:.3f}")
    print(f"Actual test prices inside interval: {interval_coverage * 100:.1f}%")

    return ForecastResult(
        dates=split.test_price_dates,
        actual_prices=split.test_prices,
        arma_return_forecast=forecast_returns,
        arma_price_forecast=arma_price_forecast,
        arma_price_lower=arma_price_lower,
        arma_price_upper=arma_price_upper,
    )


def compute_forecast_errors(
    actual_prices: npt.NDArray[np.float64],
    predicted_prices: npt.NDArray[np.float64],
    model_name: str,
) -> ForecastErrors:
    errors = actual_prices - predicted_prices
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / actual_prices)) * 100)

    return ForecastErrors(
        model_name=model_name,
        mae=mae,
        rmse=rmse,
        mape=mape,
    )


def evaluate_forecasts(forecast: ForecastResult) -> list[ForecastErrors]:
    metrics = [
        compute_forecast_errors(
            forecast.actual_prices,
            forecast.arma_price_forecast,
            "ARMA forecast",
        ),
    ]

    print("\nForecast errors on held-out prices")
    print("Model                     MAE       RMSE      MAPE")
    for metric in metrics:
        print(
            f"{metric.model_name:<24}  "
            f"{metric.mae:8.4f}  {metric.rmse:8.4f}  {metric.mape:7.2f}%"
        )

    return metrics


def plot_forecast(
    split: TrainTestSplit,
    forecast: ForecastResult,
    train_tail: int = 60,
) -> None:
    tail = min(train_tail, len(split.train_prices))
    forecast_plot_dates = np.concatenate(
        ([split.last_train_date], forecast.dates),
    )
    actual_plot_prices = np.concatenate(
        ([split.last_train_price], forecast.actual_prices),
    )
    arma_plot_prices = np.concatenate(
        ([split.last_train_price], forecast.arma_price_forecast),
    )
    arma_lower_plot_prices = np.concatenate(
        ([split.last_train_price], forecast.arma_price_lower),
    )
    arma_upper_plot_prices = np.concatenate(
        ([split.last_train_price], forecast.arma_price_upper),
    )

    plt.figure(figsize=(12, 5))
    plt.plot(
        split.train_price_dates[-tail:],
        split.train_prices[-tail:],
        label="Training prices",
        color="tab:blue",
        alpha=0.55,
    )
    plt.plot(
        forecast_plot_dates,
        actual_plot_prices,
        label="Actual test prices",
        color="black",
        linewidth=2,
    )
    plt.fill_between(
        forecast_plot_dates,
        arma_lower_plot_prices,
        arma_upper_plot_prices,
        color="tab:red",
        alpha=0.18,
        label="95% prediction interval",
    )
    plt.plot(
        forecast_plot_dates,
        arma_plot_prices,
        label="ARMA forecast",
        color="tab:red",
        linestyle="--",
        linewidth=2,
    )
    plt.axvline(split.last_train_date, color="gray", linewidth=1, alpha=0.7)
    plt.title("Propane Price Forecast")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({PRICE_UNITS})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def check_stationarity(
    raw_prices: npt.NDArray[np.float64],
    log_prices: npt.NDArray[np.float64],
    log_returns: npt.NDArray[np.float64],
    alpha: float = 0.05,
    acf_lags: int = 40,
) -> dict[str, bool]:
    stationarity_results = {}
    adf_series = {
        "Raw observed prices": raw_prices[np.isfinite(raw_prices) & (raw_prices > 0)],
        "Log prices": log_prices[np.isfinite(log_prices)],
        "Log returns": log_returns[np.isfinite(log_returns)],
    }

    print("\nStationarity checks using the ADF test")
    print("H0: the series has a unit root and is non-stationary.")

    for series_name, series in adf_series.items():
        if len(series) < 3:
            raise ValueError(f"{series_name} is too short for an ADF test.")

        adf_statistic, p_value, used_lag, observations, critical_values, _ = adfuller(
            series,
            autolag="AIC",
        )
        is_stationary = p_value < alpha
        stationarity_results[series_name] = is_stationary

        print(f"\n{series_name}")
        print(f"ADF statistic: {adf_statistic:.6f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Used lag: {used_lag}")
        print(f"Observations: {observations}")
        for level, value in critical_values.items():
            print(f"Critical value ({level}): {value:.6f}")
        print(f"Reject H0 at alpha={alpha}: {is_stationary}")

    valid_returns = adf_series["Log returns"]
    lags = min(acf_lags, len(valid_returns) // 2 - 1)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(valid_returns, ax=ax1, lags=lags)
    ax1.set_title("ACF of Propane Log Returns")
    plot_pacf(valid_returns, ax=ax2, lags=lags, method="ywm")
    ax2.set_title("PACF of Propane Log Returns")
    plt.tight_layout()
    plt.show()

    return stationarity_results


def plot_raw_data(
    dates: npt.NDArray[np.datetime64], prices: npt.NDArray[np.float64]
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, prices)
    plt.title("Raw Propane Prices")
    plt.xlabel("Date")
    plt.ylabel(f"Price ({PRICE_UNITS})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cleaned_data(
    dates: npt.NDArray[np.datetime64], log_returns: npt.NDArray[np.float64]
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(dates, log_returns, linewidth=1)
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.title("Cleaned Propane Log Returns")
    plt.xlabel("Date")
    plt.ylabel("Log return")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    dates, raw_prices = load_propane()
    plot_raw_data(dates, raw_prices)

    clean_dates, clean_prices, log_prices, return_dates, log_returns = clean_data(
        dates,
        raw_prices,
    )

    plot_cleaned_data(return_dates, log_returns)
    check_stationarity(raw_prices, log_prices, log_returns)
    split = make_train_test_split(clean_dates, clean_prices, return_dates, log_returns)

    if RUN_GRID_SEARCH:
        grid_search = grid_search_arma(split.train_returns)
        selected_model = grid_search.best_result
    else:
        p, q = HARDCODED_ARMA_ORDER
        selected_model = ARMAGridResult(
            p=p,
            q=q,
            aic=np.nan,
            bic=np.nan,
            converged=True,
        )
        print(f"\nSkipping grid search; using hardcoded ARMA({p}, {q}) model.")

    fitted_model = fit_final_arma_model(split.train_returns, selected_model)
    forecast = forecast_prices(fitted_model, split)
    plot_forecast(split, forecast)
    evaluate_forecasts(forecast)


if __name__ == "__main__":
    main()
