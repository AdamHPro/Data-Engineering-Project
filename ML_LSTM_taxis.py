import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Nettoyage des données
def clean_data(df):
    df['month'] = pd.to_datetime(df['month'])
    df_agg = df.groupby('month')['taxi_fleet'].sum().reset_index()
    full_dates = pd.date_range(start=df_agg['month'].min(), end=df_agg['month'].max(), freq='MS')
    df_agg = df_agg.set_index('month').reindex(full_dates).reset_index()
    df_agg.columns = ['month', 'taxi_fleet']
    df_agg['taxi_fleet'] = df_agg['taxi_fleet'].interpolate()
    return df_agg

# Chargement et prétraitement des données
df = pd.read_csv('monthly_taxi_fleet.csv')
df_clean = clean_data(df)

# Normalisation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_clean[['taxi_fleet']])

# Paramètres du modèle
TEST_SIZE = 0.2

# Division train-test
split = int(len(scaled_data) * (1 - TEST_SIZE))
train, test = scaled_data[:split], scaled_data[split:]

# Modèle ARIMA
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Prédictions
train_predict = model_fit.predict(start=0, end=len(train)-1)
test_predict = model_fit.forecast(steps=len(test))

# Inversion de la normalisation
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
train_actual = scaler.inverse_transform(train)
test_actual = scaler.inverse_transform(test)

# Calcul des métriques
print(f"Train RMSE: {np.sqrt(mean_squared_error(train_actual, train_predict))}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(test_actual, test_predict))}")
print(f"Test MAE: {mean_absolute_error(test_actual, test_predict)}")

# Visualisation
plt.figure(figsize=(15,6))
plt.plot(df_clean['month'], df_clean['taxi_fleet'], label='Données réelles')
plt.plot(df_clean['month'][:split], train_predict, label='Prédictions (train)')
plt.plot(df_clean['month'][split:], test_predict, label='Prédictions (test)')
plt.legend()
plt.title('Performance du modèle')
plt.show()

# Prédiction jusqu'en 2030
future_dates = pd.date_range(start=df_clean['month'].iloc[-1] + pd.DateOffset(months=1), end='2030-12-01', freq='MS')
future_predict = model_fit.forecast(steps=len(future_dates))
future_predict = scaler.inverse_transform(future_predict.reshape(-1, 1))

# Visualisation finale
plt.figure(figsize=(15,6))
plt.plot(df_clean['month'], df_clean['taxi_fleet'], label='Données historiques')
plt.plot(future_dates, future_predict, label='Prédictions jusqu\'en 2030', linestyle='--')
plt.axvline(x=df_clean['month'].iloc[-1], color='gray', linestyle='--')
plt.title('Prévision de la flotte de taxis jusqu\'en 2030')
plt.xlabel('Date')
plt.ylabel('Nombre de taxis')
plt.legend()
plt.show()
