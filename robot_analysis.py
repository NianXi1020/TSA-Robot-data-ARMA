"""Time series analysis for the industrial robot dataset."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf


def load_robot_series(file_path: Path) -> pd.Series:
    """Load the robot time series from the provided path."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    series = pd.read_csv(file_path, header=0, names=["robot"])
    return series["robot"].astype(float)


# %% (a) Time series plot and stationarity assessment
robot_path = Path("robot.dat")
robot_series = load_robot_series(robot_path)

plt.figure(figsize=(12, 4))
plt.plot(robot_series.index + 1, robot_series.values, color="steelblue")
plt.title("Robot Position Error Time Series")
plt.xlabel("Observation")
plt.ylabel("Distance from Target (inches)")
plt.tight_layout()
plt.savefig("robot_time_series.png", dpi=300)
plt.show()

midpoint = len(robot_series) // 2
mean_shift = abs(robot_series.iloc[:midpoint].mean() - robot_series.iloc[midpoint:].mean())
threshold = robot_series.std() * 0.1
if mean_shift < threshold:
    stationarity_comment = "The time series looks roughly stationary based on the plot."
else:
    stationarity_comment = "The time series shows noticeable level shifts, suggesting nonstationarity."
print(stationarity_comment)


# %% (b) Sample ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(robot_series, lags=40, ax=axes[0])
plot_pacf(robot_series, lags=40, ax=axes[1], method="ywm")
axes[0].set_title("Sample ACF of Robot Series")
axes[1].set_title("Sample PACF of Robot Series")
plt.tight_layout()
fig.savefig("robot_acf_pacf.png", dpi=300)
plt.show()

print("Use the ACF and PACF to judge whether the original series is stationary or requires differencing.")


# %% (c) Augmented Dickey-Fuller test
adf_result = adfuller(robot_series, autolag="AIC")
adf_output = pd.Series(
    {
        "ADF Statistic": adf_result[0],
        "p-value": adf_result[1],
        "Used Lags": adf_result[2],
        "Number of Observations": adf_result[3],
    }
)
print(adf_output)
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

if adf_result[1] < 0.05:
    print("Reject the null hypothesis of a unit root: the series is likely stationary.")
else:
    print("Fail to reject the null hypothesis: the series is likely nonstationary.")


# %% (d) Difference the series and plot ACF/PACF
robot_diff = robot_series.diff().dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(robot_diff, lags=40, ax=axes[0])
plot_pacf(robot_diff, lags=40, ax=axes[1], method="ywm")
axes[0].set_title("Sample ACF of Differenced Robot Series")
axes[1].set_title("Sample PACF of Differenced Robot Series")
plt.tight_layout()
fig.savefig("robot_diff_acf_pacf.png", dpi=300)
plt.show()

midpoint_diff = len(robot_diff) // 2
mean_shift_diff = abs(robot_diff.iloc[:midpoint_diff].mean() - robot_diff.iloc[midpoint_diff:].mean())
threshold_diff = robot_diff.std() * 0.1
if mean_shift_diff < threshold_diff:
    diff_comment = "The differenced series now looks stationary."
else:
    diff_comment = "The differenced series still shows signs of nonstationarity."
print(diff_comment)


# %% (e) Extended autocorrelation function (EACF) of differenced series
def compute_eacf(series: pd.Series, ar_max: int = 6, ma_max: int = 6) -> pd.DataFrame:
    """Compute a simple EACF table using residual autocorrelations."""
    series = series - series.mean()
    eacf_values = np.zeros((ar_max + 1, ma_max + 1))

    for p in range(ar_max + 1):
        if p == 0:
            residuals = series
        else:
            model = AutoReg(series, lags=p, old_names=False).fit()
            residuals = model.resid
        residuals = residuals - residuals.mean()
        acf_vals = acf(residuals, nlags=ma_max + 1, fft=False)
        eacf_values[p, :] = np.abs(acf_vals[1 : ma_max + 2])

    ar_index = [f"AR({p})" for p in range(ar_max + 1)]
    ma_columns = [f"MA({q})" for q in range(ma_max + 1)]
    return pd.DataFrame(eacf_values, index=ar_index, columns=ma_columns)


eacf_table = compute_eacf(robot_diff, ar_max=6, ma_max=6)
print(eacf_table)

min_position = np.unravel_index(np.argmin(eacf_table.values), eacf_table.shape)
print(
    "Smallest absolute residual autocorrelation occurs near "
    f"AR order {min_position[0]} and MA order {min_position[1]}."
)


# %% (f) Best subsets ARMA selection on differenced series
def best_subset_arma(series: pd.Series, max_p: int = 5, max_q: int = 5) -> pd.DataFrame:
    """Evaluate ARMA(p, q) models and rank by information criteria."""
    results = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(series, order=(p, 0, q)).fit(method="innovations_mle")
                results.append(
                    {
                        "p": p,
                        "q": q,
                        "AIC": model.aic,
                        "BIC": model.bic,
                        "HQIC": model.hqic,
                    }
                )
            except Exception:
                continue
    if not results:
        raise RuntimeError("No ARMA models were successfully estimated.")
    result_df = pd.DataFrame(results)
    return result_df.sort_values("AIC").reset_index(drop=True)


arma_results = best_subset_arma(robot_diff, max_p=5, max_q=5)
print(arma_results.head(10))

best_model = arma_results.iloc[0]
print(
    f"Best subset ARMA model based on AIC is ARMA({int(best_model['p'])}, {int(best_model['q'])})."
)
print("Compare this selection with earlier diagnostics to confirm consistency.")
