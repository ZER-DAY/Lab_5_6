# Importing required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to display missing values
def missing_values_table(df):
    # حساب القيم المفقودة ونسبتها
    # Calculate missing values and their percentage
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    # إنشاء جدول للقيم المفقودة
    # Create a table of missing values
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # إعادة تسمية الأعمدة في الجدول
    # Rename columns in the table
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    
    # تصفية الأعمدة ذات القيم المفقودة وترتيبها تنازلياً
    # Filter columns with missing values and sort them in descending order
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0
    ].sort_values('% of Total Values', ascending=False).round(1)
    
    # طباعة ملخص عدد الأعمدة والقيم المفقودة
    # Print summary of the number of columns and missing values
    print(
        "Your selected dataframe has " + str(df.shape[1]) + " columns.\n" +
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
        " columns that have missing values."
    )
    
    # إرجاع الجدول النهائي
    # Return the final table
    return mis_val_table_ren_columns

# قراءة بيانات ملف CSV
# Read the CSV file
df = pd.read_csv("C:/Users/Bahaa/Desktop/Study/FundamentalsofMathematics/Lab5%6/.venv/litov_test_mod.csv")

# تحديد الأعمدة المطلوبة
# Columns of interest
columns_of_interest = [
    "Я трачу на учебу такое кол-во часов в неделю:",  # عدد الساعات التي يقضيها المستخدم في الدراسة
    "Я трачу на учебу такой % моего времени за неделю:",  # النسبة المئوية من الوقت المخصص للدراسة
    "Я отдыхаю такое кол-во часов в неделю:",  # عدد ساعات الراحة في الأسبوع
    "Курс",  # الدورة أو الفصل الدراسي
]

df_select = df[columns_of_interest]

# الأعمدة العددية لمعالجة القيم المفقودة
# Numeric columns to handle missing values
numeric_columns = [
    "Я трачу на учебу такое кол-во часов в неделю:",
    "Я трачу на учебу такой % моего времени за неделю:",
    "Я отдыхаю такое кол-во часов в неделю:"
]

# ملء القيم المفقودة بمتوسط القيم
# Fill missing values with mean
for col in numeric_columns:
    mean_value = df_select[col].mean()  
    df_select[col] = df_select[col].fillna(mean_value)  

# تصحيح القيم التي تتجاوز الحدود المسموح بها
# Replace values exceeding specified limits
df_select["Я трачу на учебу такое кол-во часов в неделю:"] = df_select[
    "Я трачу на учебу такое кол-во часов в неделю:"
].apply(lambda x: min(x, 110))

df_select["Я трачу на учебу такой % моего времени за неделю:"] = df_select[
    "Я трачу на учебу такой % моего времени за неделю:"
].apply(lambda x: min(x, 80))

df_select["Я отдыхаю такое кол-во часов в неделю:"] = df_select[
    "Я отдыхаю такое кол-во часов в неделю:"
].apply(lambda x: min(x, 90))

# حساب مصفوفة الارتباط
# Compute correlation matrix
correlation_matrix = df_select[numeric_columns].corr()

# رسم مصفوفة الارتباط
# Plot correlation matrix
plt.figure(figsize=(8, 6)) 
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap="coolwarm", 
    fmt=".2f", 
    linewidths=0.5, 
    cbar_kws={'label': 'Correlation'}
)  
plt.title("Матрица корреляции", fontsize=16)  # عنوان الرسم البياني
plt.xticks(rotation=45, ha='right', fontsize=10) 
plt.yticks(fontsize=10)
plt.tight_layout()

# إنشاء رسوم بيانية باستخدام Boxplot
# Create box plots for numeric columns by course
for column in numeric_columns:
    plt.figure(figsize=(10, 6))  
    sns.boxplot(data=df_select, x="Курс", y=column, palette="muted")
    plt.title(f"Распределение: {column} по курсам", fontsize=16)  # العنوان
    plt.xlabel("Курс", fontsize=12)  # تسميات المحاور
    plt.ylabel(column, fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)  
    
plt.show()  # عرض جميع الرسوم البيانية
