import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.title("🔮 Smart Grid Load Forecasting using Prophet")

# Load dataset
data = pd.read_csv("PJME_hourly.csv", parse_dates=["Datetime"])
data = data.loc[data["Datetime"].dt.month == 1]
data = data[["Datetime", "PJME_MW"]].rename(columns={"Datetime": "ds", "PJME_MW": "y"})

# Fit Prophet
model = Prophet()
model.fit(data)

# Forecast 24 hours
future = model.make_future_dataframe(periods=24, freq="H")
forecast = model.predict(future)

# Plot
st.subheader("📊 Prophet Forecast")
st.pyplot(model.plot(forecast))

# Compare with actual
actual = data.tail(24)["y"].values
pred = forecast.tail(24)["yhat"].values
error = actual - pred
anomalies = np.where(np.abs(error) > 1000)[0]

# Custom Plot
st.subheader("⚠️ Anomalies in Forecast")
fig, ax = plt.subplots()
ax.plot(pred, label="Forecast", marker="o")
ax.plot(actual, label="Actual", marker="x")
ax.scatter(anomalies, actual[anomalies], color="red", label="Anomaly", zorder=5)
ax.set_title("Actual vs Forecast - Last 24 Hours")
ax.legend()
st.pyplot(fig)
