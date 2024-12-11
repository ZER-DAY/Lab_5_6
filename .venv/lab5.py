# Importing the required libraries
# استيراد المكتبات المطلوبة
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Loading the dataset and renaming columns
# تحميل البيانات وإعادة تسمية الأعمدة
df = pd.read_csv("C:/Users/Bahaa/Desktop/Study/FundamentalsofMathematics/Lab5%6/.venv/AirPassengers.csv")
df.columns = ['Month', 'Passengers']
df['Month'] = pd.to_datetime(df['Month'])  # Converting 'Month' to datetime
# تحويل العمود "Month" إلى نوع بيانات التاريخ
df.set_index('Month', inplace=True)  # Setting the 'Month' column as the index
# تعيين عمود "Month" كفهرس

# Visualizing the time series data
# عرض البيانات الزمنية
plt.figure(figsize=(10, 6))
plt.plot(df, label='Количество пассажиров')
plt.title("Количество авиапассажиров по месяцам")  # Title in Russian
# العنوان مكتوب باللغة الروسية
plt.xlabel("Дата")  # X-axis label in Russian
# عنوان المحور X باللغة الروسية
plt.ylabel("Количество пассажиров")  # Y-axis label in Russian
# عنوان المحور Y باللغة الروسية
plt.legend()
plt.show()

# Dickey-Fuller test for stationarity
# اختبار ديكي-فولر للتحقق من استقرار السلسلة الزمنية
result = adfuller(df['Passengers'])
print("Результаты теста Дики-Фуллера:")  # Printing results in Russian
# طباعة النتائج باللغة الروسية
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# 1. Applying logarithm transformation to the time series
# 1. تطبيق التحويل اللوغاريتمي على السلسلة الزمنية
df_log = np.log(df['Passengers'])

# 2. Calculating moving average with a 12-month window and subtracting it
# 2. حساب المتوسط المتحرك بفترة 12 شهرًا وطرحه
moving_avg = df_log.rolling(window=12).mean()
df_log_moving_avg_diff = df_log - moving_avg
df_log_moving_avg_diff.dropna(inplace=True)  # Removing NaN values
# إزالة القيم غير المعرفة (NaN)

# Visualizing the transformed series
# عرض السلسلة الزمنية بعد التحويل
plt.figure(figsize=(10, 6))
plt.plot(df_log_moving_avg_diff, label='Логарифмированный ряд - Скользящее среднее')  
# Label in Russian
# التسمية باللغة الروسية
plt.title("Преобразованный временной ряд (логарифмированный и вычтенное скользящее среднее)")
# العنوان مكتوب باللغة الروسية
plt.xlabel("Дата")  # X-axis label in Russian
# عنوان المحور X باللغة الروسية
plt.ylabel("Разница логарифмов")  # Y-axis label in Russian
# عنوان المحور Y باللغة الروسية
plt.legend()
plt.show()

# 3. Building an ARIMA model on the transformed series
# 3. بناء نموذج ARIMA على السلسلة الزمنية بعد التحويل
model = ARIMA(df_log_moving_avg_diff, order=(1, 1, 1))  # Parameters (p=1, d=1, q=1) can be adjusted
# يمكن تعديل المعاملات (p=1, d=1, q=1)
model_fit = model.fit()

# 4. Forecasting for the next 10 years (120 months)
# 4. التنبؤ للأعوام العشرة المقبلة (120 شهرًا)
forecast_steps = 120
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Transforming the forecast back to the original scale
# تحويل التنبؤ إلى النطاق الأصلي
forecast_values = forecast.predicted_mean + moving_avg[-1]
forecast_values = np.exp(forecast_values)

# Visualizing the original series and forecast
# عرض السلسلة الأصلية والتنبؤ
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Исходные данные')  # Original data
# البيانات الأصلية
plt.plot(forecast_index, forecast_values, color='red', label='Прогноз ARIMA на 10 лет')  
# Forecast label in Russian
# التسمية باللغة الروسية
plt.title("Прогноз количества авиапассажиров на 10 лет вперед")  # Title in Russian
# العنوان مكتوب باللغة الروسية
plt.xlabel("Дата")  # X-axis label in Russian
# عنوان المحور X باللغة الروسية
plt.ylabel("Количество пассажиров")  # Y-axis label in Russian
# عنوان المحور Y باللغة الروسية
plt.legend()
plt.show()

# Building a SARIMA model on the log-transformed data considering seasonality
# بناء نموذج SARIMA على البيانات اللوغاريتمية مع مراعاة التأثير الموسمي
model = SARIMAX(df_log_moving_avg_diff, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Forecasting for the next 10 years (120 months)
# التنبؤ للأعوام العشرة المقبلة (120 شهرًا)
forecast_steps = 120
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='M')[1:]

# Transforming the SARIMA forecast back to the original scale
# تحويل تنبؤ SARIMA إلى النطاق الأصلي
forecast_values = forecast.predicted_mean + moving_avg[-1]
forecast_values = np.exp(forecast_values)

# Visualizing the original series and SARIMA forecast
# عرض السلسلة الأصلية وتنبؤ SARIMA
plt.figure(figsize=(12, 6))
plt.plot(df['Passengers'], label='Исходные данные')  # Original data
# البيانات الأصلية
plt.plot(forecast_index, forecast_values, color='red', label='Прогноз SARIMA на 10 лет')  
# SARIMA forecast label in Russian
# التسمية باللغة الروسية
plt.title("Прогноз количества авиапассажиров на 10 лет вперед (SARIMA)")  # Title in Russian
# العنوان مكتوب باللغة الروسية
plt.xlabel("Дата")  # X-axis label in Russian
# عنوان المحور X باللغة الروسية
plt.ylabel("Количество пассажиров")  # Y-axis label in Russian
# عنوان المحور Y باللغة الروسية
plt.legend()
plt.show()
