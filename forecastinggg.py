# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# =========================================
# 2. LOAD DATA
# =========================================
data = pd.read_csv("retail_sales.csv")   # make sure file is in same folder

print(data.head())

# =========================================
# 3. PREPROCESSING
# =========================================
# Convert date
data['date'] = pd.to_datetime(data['date'])

# Sort
data = data.sort_values('date')

# Set index
data.set_index('date', inplace=True)

# Convert to single time series (VERY IMPORTANT)
data_ts = data.groupby('date')['sales'].sum()

print(data_ts.head())

# =========================================
# 4. VISUALIZATION
# =========================================
plt.figure(figsize=(10,5))
plt.plot(data_ts)
plt.title("Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# =========================================
# 5. AUTO ARIMA MODEL
# =========================================
auto_model = auto_arima(
    data_ts,
    seasonal=True,
    m=7,              # weekly seasonality
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

print(auto_model.summary())

# =========================================
# 6. BUILD FINAL ARIMA MODEL
# =========================================
# Get best parameters
p, d, q = auto_model.order

model = ARIMA(data_ts, order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

# =========================================
# 7. FORECAST FUTURE
# =========================================
forecast = model_fit.forecast(steps=30)

print("\nNext 30 Days Forecast:")
print(forecast)

# =========================================
# 8. PLOT FORECAST
# =========================================
plt.figure(figsize=(10,5))
plt.plot(data_ts, label="Actual")
plt.plot(forecast, label="Forecast", color='red')
plt.legend()
plt.title("Sales Forecast")
plt.show()
# =========================================
# 9. REGRESSION MODEL (Linear Regression)
# =========================================
from sklearn.linear_model import LinearRegression

# Create time index
data_ts = data_ts.reset_index()
data_ts['time'] = range(len(data_ts))

# Features and target
X = data_ts[['time']]
y = data_ts['sales']

# Train-test split
train_size = int(len(data_ts) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict
y_pred = lr_model.predict(X_test)

# =========================================
# 10. EVALUATE REGRESSION
# =========================================
from sklearn.metrics import mean_absolute_error

mae_lr = mean_absolute_error(y_test, y_pred)
print("Linear Regression MAE:", mae_lr)

# =========================================
# 11. PLOT COMPARISON
# =========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Regression Prediction", color='green')
plt.legend()
plt.title("Regression vs Actual")
plt.show()
from sklearn.metrics import mean_absolute_error

# Split data
train = data_ts[:-30]
test = data_ts[-30:]

# Train ARIMA
model = ARIMA(train['sales'], order=(p, d, q))
model_fit = model.fit()

# Predict
pred = model_fit.forecast(steps=30)

# Evaluate
mae_arima = mean_absolute_error(test['sales'], pred)

print("ARIMA MAE:", mae_arima)
# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# 2. LOAD DATA
# =========================
data = pd.read_csv("retail_sales.csv")

# keep only sales column
data = data[['sales']]

# =========================
# 3. SCALE DATA (0 to 1)
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================
# 4. CREATE SEQUENCES
# =========================
X = []
y = []

window_size = 60

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# reshape for LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# =========================
# 5. BUILD LSTM MODEL
# =========================
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# =========================
# 6. TRAIN MODEL
# =========================
model.fit(X, y, epochs=5, batch_size=32)

# =========================
# 7. PREDICTION
# =========================
predictions = model.predict(X)

# =========================
# 8. INVERSE TRANSFORM (IMPORTANT)
# =========================
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y.reshape(-1, 1))

# =========================
# 9. PLOT RESULTS
# =========================
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual Sales")
plt.plot(predictions, label="LSTM Prediction")
plt.title("LSTM Sales Forecasting")
plt.legend()
plt.show()

# =========================
# 10. ERROR (MAE)
# =========================
mae = mean_absolute_error(actual, predictions)
print("LSTM MAE:", mae)