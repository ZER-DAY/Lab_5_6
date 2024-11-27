import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

def handle_missing_values(df):
    # Удаление столбцов с более чем 80% пропусков
    threshold = 0.8 * len(df)  # 80%
    df.dropna(axis=1, thresh=threshold, inplace=True)
    
def fill_missing_values(df):
    for column in df.columns:
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':  # Числовые столбцы
            df[column] = df[column].fillna(df[column].mean())
        elif df[column].dtype == 'object':  # Текстовые/категориальные столбцы
            df[column] = df[column].fillna("Unknown")

df = pd.read_csv(".venv/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv")

df.replace("Not Available", np.nan, inplace=True)

handle_missing_values(df)
fill_missing_values(df)

for column in df.select_dtypes(include="int64").columns:
    # Округляем значения и преобразуем в int64
    df[column] = df[column].round(0).astype('float64')


numeric_columns = [
    'ENERGY STAR Score', 'DOF Gross Floor Area', 'Year Built', 
    'Occupancy', 'Total GHG Emissions (Metric Tons CO2e)', 'Site EUI (kBtu/ft²)'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Создаем подмножество данных с переименованными колонками
df_subset = df[['ENERGY STAR Score', 'DOF Gross Floor Area', 'Year Built', 
                'Occupancy', 'Total GHG Emissions (Metric Tons CO2e)', 'Site EUI (kBtu/ft²)']].copy()


df_subset.columns = [
    'Рейтинг ENERGY STAR', 'Общая площадь (фут²)', 'Год постройки',
    'Занятость (%)', 'Выбросы CO2 (тонн)', 'Энергопотребление (kBtu/ft²)'
]

# Построение PairPlot
sns.pairplot(df_subset, diag_kind='kde')
plt.suptitle("Зависимости между переменными, влияющими на рейтинг энергопотребления", y=1.02)
plt.show()


numeric_columns = df.select_dtypes(include=[np.number]).columns

df[numeric_columns] = df[numeric_columns].apply(np.log1p)


numeric_df = df.select_dtypes(include=[np.number])

corr_matrix = numeric_df.corr()

print("Корреляционная матрица:")
print(corr_matrix)

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

threshold = 0.6

columns_to_keep = [column for column in upper.columns if any(upper[column] > threshold)]

print("\nКолонки с высокой корреляцией (больше 0.6), которые будут сохранены:")
print(columns_to_keep)

df_reduced = df[columns_to_keep]

print("\nПосле выбора признаков с высокой корреляцией:")
print(df_reduced.info())

print("\nТаблица пропущенных значений:")
print(missing_values_table(df_reduced))


df_cleaned = df_reduced.dropna(axis=1, how='any')

print("\nПосле удаления колонок с пропущенными значениями:")
print(df_cleaned.info())
print(missing_values_table(df_cleaned))

# Целевая переменная (target) - выберем 'Property GFA - Self-Reported (ft²)' как целевую переменную
y = df_cleaned['Property GFA - Self-Reported (ft²)']

# Признаки (features) - выберем 'Council District' как признак
X = df_cleaned[['Council District']]

# Разделяем данные на обучающую и тестовую выборки (80% обучение, 20% тест)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализируем модели
models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor (SVR)': SVR(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'K-Neighbors Regressor': KNeighborsRegressor(n_neighbors=5)
}

# Оценка MAE для каждой модели
mae_scores = {}

for model_name, model in models.items():
    # Обучаем модель
    model.fit(X_train, y_train)
    
    # Делаем прогноз на тестовой выборке
    y_pred = model.predict(X_test)
    
    # Вычисляем MAE
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores[model_name] = mae

# Выводим результаты
print("Среднее абсолютное отклонение (MAE) для каждой модели:")
for model_name, mae in mae_scores.items():
    print(f"{model_name}: {mae:.4f}")