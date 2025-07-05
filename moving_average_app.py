import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("📉 Smart Grid Forecasting (Moving Average)")

data = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"], index_col="Datetime")
data = data.loc["2017-01"]
series = data["PJME_MW"].resample("H").mean().ffill()

def moving_avg_forecast(series, days=7):
    forecast = []
    for hour in range(24):
        values = []
        for d in range(1, days + 1):
            idx = series.index[-1] - pd.Timedelta(days=d) + pd.Timedelta(hours=hour)
            if idx in series.index:
                values.append(series[idx])
        forecast.append(np.mean(values))
    return forecast

forecast = moving_avg_forecast(series)
actual = series[-24:]

st.subheader("📊 24-Hour Forecast vs Actual")
fig, ax = plt.subplots()
ax.plot(forecast, label="Forecast", marker="o")
ax.plot(actual.values, label="Actual", marker="x")
ax.set_title("Moving Average Load Forecast")
ax.legend()
st.pyplot(fig)
