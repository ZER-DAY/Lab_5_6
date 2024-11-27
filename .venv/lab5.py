import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv(".venv/AirPassengers.csv")
df.columns = ['Month', 'Passengers']
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Визуализация временного ряда
# plt.figure(figsize=(10, 6))
# plt.plot(df, label='Количество пассажиров')
# plt.title("Количество авиапассажиров по месяцам")
# plt.xlabel("Дата")
# plt.ylabel("Количество пассажиров")
# plt.legend()
# plt.show()

# Тест Дики-Фуллера
result = adfuller(df['Passengers'])
print("Результаты теста Дики-Фуллера:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# 1. Логарифмируем временной ряд
df_log = np.log(df['Passengers'])

# 2. Вычислим скользящее среднее с окном 12 месяцев и вычтем его из логарифмированного ряда
moving_avg = df_log.rolling(window=12).mean()
df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)  # Удаляем NaN значения

# Визуализация преобразованного ряда
plt.figure(figsize=(10, 6))
plt.plot(df_log_moving_avg_diff, label='Логарифмированный ряд - Скользящее среднее')
plt.title("Преобразованный временной ряд (логарифмированный и вычтенное скользящее среднее)")
plt.xlabel("Дата")
plt.ylabel("Разница логарифмов")
plt.legend()
plt.show()

# 3. Построение модели ARIMA на преобразованном ряду
model = ARIMA(df_log_moving_avg_diff, order=(1, 1, 1))  # Параметры (p=1, d=1, q=1) можно менять
model_fit = model.fit()

# 4. Прогнозирование на следующие 10 лет (120 месяцев)
forecast_steps = 120
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Преобразуем прогноз к исходному масштабу, добавив скользящее среднее и возвратив логарифм
forecast_values = forecast.predicted_mean + moving_avg[-1]
forecast_values = np.exp(forecast_values)

# Визуализация исходного ряда и прогноза
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Исходные данные')
plt.plot(forecast_index, forecast_values, color='red', label='Прогноз ARIMA на 10 лет')
plt.title("Прогноз количества авиапассажиров на 10 лет вперед")
plt.xlabel("Дата")
plt.ylabel("Количество пассажиров")
plt.legend()
plt.show()

# # Построение модели SARIMA на логарифмированных данных с учетом сезонности
model = SARIMAX(df_log_moving_avg_diff, order=(1, 1
                                               , 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Прогнозирование на следующие 10 лет (120 месяцев)
forecast_steps = 120
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Обратное преобразование прогноза
forecast_values = forecast.predicted_mean + moving_avg[-1]
forecast_values = np.exp(forecast_values)

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Исходные данные')
plt.plot(forecast_index, forecast_values, color='red', label='Прогноз SARIMA на 10 лет')
plt.title("Прогноз количества авиапассажиров на 10 лет вперед (SARIMA)")
plt.xlabel("Дата")
plt.ylabel("Количество пассажиров")
plt.legend()
plt.show()