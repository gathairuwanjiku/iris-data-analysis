import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the dataset
iris_data = load_iris(as_frame=True)
df = iris_data.frame

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistics:")
print(df.describe())

# Add species names
df['species'] = df['target'].map(dict(enumerate(iris_data.target_names)))

# Group by species
print("\nMean by species:")
print(df.groupby('species').mean())

# ----------- PLOTS -----------

# Line chart
df.groupby('species').mean()['petal length (cm)'].plot(kind='line', marker='o', title='Mean Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.show()

# Bar chart
df.groupby('species').mean()['sepal width (cm)'].plot(kind='bar', color='skyblue', title='Mean Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Histogram
df['sepal length (cm)'].plot(kind='hist', bins=20, color='purple', title='Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.show()

# Scatter plot
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()