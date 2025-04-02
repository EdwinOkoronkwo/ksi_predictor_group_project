"""
Created on Sat Mar 15 16:22:00 2025

@author: oesez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''Step 1:  Data exploration'''
# Load the dataset
df = pd.read_csv("../data/total_ksi.csv")

# Display basic info
print("Basic Information:\n",df.info())

# Display summary statistics
summary_stats = df.describe()
print("\nSummary Statistics:\n", summary_stats)

missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Visualize missing data
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap ")
plt.show()

'''Step 2: Statistical Analysis'''
# Correlation matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Spectral', fmt='.2f')
plt.title("Correlation Matrix for Numeric Columns")
plt.show()

# Distribution of numerical variables
df.hist(bins=30, figsize=(15, 10))
plt.title("Numerical variables")
plt.show()

'''Step 3: Handle Missing Data'''
# Select numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Fill missing values with the mean for numerical columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Drop columns with too many missing values
df.dropna(axis=1, thresh=0.7 * len(df), inplace=True)