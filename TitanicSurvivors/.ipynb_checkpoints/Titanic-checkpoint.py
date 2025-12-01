import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Replace 'your_file.csv' with your file path
df = pd.read_csv("TitanicSurvivors/train.csv")

for col in df.columns:
    missing_percent = df[col].isnull().mean() * 100
    print(f"{missing_percent:.2f}% missing in {col}")
# Histograms for each feature
df.hist(figsize=(8,6))
plt.suptitle('Feature Distributions')
plt.show()

df.info()
df.describe()
df.isnull().sum()
df['Survived'].value_counts()

df['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

df['Sex'].value_counts().plot(kind='bar')
plt.title('Sex Distribution')
plt.show()

df['Pclass'].value_counts().plot(kind='bar')
plt.title('Passenger Class Distribution')
plt.show()

survived_by_sex = df.groupby('Sex')['Survived'].mean()
survived_by_sex.plot(kind='bar')
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Probability')
plt.show()
plt.scatter(df['Age'], df['Survived'])
plt.title('Age vs Survival')
plt.xlabel('Age')
plt.ylabel('Survived (1=Yes, 0=No)')
plt.show()

corr = df.corr(numeric_only=True)
plt.imshow(corr, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Feature Correlation Matrix')
plt.show()
