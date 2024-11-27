import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

df = pd.read_csv(".venv/litov_test_mod.csv")

columns_of_interest = [
    "Я трачу на учебу такое кол-во часов в неделю:",
    "Я трачу на учебу такой % моего времени за неделю:",
    "Я отдыхаю такое кол-во часов в неделю:",
    "Курс",
]

df_select = df[columns_of_interest]

numeric_columns = [
    "Я трачу на учебу такое кол-во часов в неделю:",
    "Я трачу на учебу такой % моего времени за неделю:",
    "Я отдыхаю такое кол-во часов в неделю:"
]

for col in numeric_columns:
    mean_value = df_select[col].mean()  
    df_select[col] = df_select[col].fillna(mean_value)  


# Замена значений, превышающих указанные пределы
df_select["Я трачу на учебу такое кол-во часов в неделю:"] = df_select[
    "Я трачу на учебу такое кол-во часов в неделю:"
].apply(lambda x: min(x, 110))


df_select["Я трачу на учебу такой % моего времени за неделю:"] = df_select[
    "Я трачу на учебу такой % моего времени за неделю:"
].apply(lambda x: min(x, 80))

df_select["Я отдыхаю такое кол-во часов в неделю:"] = df_select[
"Я отдыхаю такое кол-во часов в неделю:"].apply(lambda x: min(x, 90))


correlation_matrix = df_select[numeric_columns].corr()

# Матрица корреляций
plt.figure(figsize=(8, 6)) 
sns.heatmap(correlation_matrix, annot=True,cmap="coolwarm", fmt=".2f",linewidths=0.5,cbar_kws={'label': 'Correlation'})  
plt.title("Матрица корреляции", fontsize=16)  
plt.xticks(rotation=45, ha='right', fontsize=10) 
plt.yticks(fontsize=10)
plt.tight_layout()  

#Графики
for column in numeric_columns:
    plt.figure(figsize=(10, 6))  
    sns.boxplot(data=df_select, x="Курс", y=column, palette="muted")
    plt.title(f"Распределение: {column} по курсам", fontsize=16)
    plt.xlabel("Курс", fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)  
    
plt.show()  