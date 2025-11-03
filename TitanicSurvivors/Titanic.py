import pandas as pd

# Replace 'your_file.csv' with your file path
df = pd.read_csv('gender_submission.csv')

# Preview the first 5 rows
print(df.head())
# Check data types
print(df.info())

# Summary statistics for numeric columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Quick look at unique values in a column
print(df['column_name'].value_counts())
