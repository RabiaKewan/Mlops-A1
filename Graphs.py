import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset into a pandas DataFrame
iris_df = pd.read_csv("iris.csv")

# Add some missing values and outliers for demonstration
iris_df.loc[10:15, 'sepal_length'] = np.nan
iris_df.loc[30:35, 'petal_width'] = 8.0

# Visualize the distribution of the numeric columns before handling missing values and outliers
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(data=iris_df, x='sepal_length', kde=True)
plt.title('Distribution of Sepal Length')

plt.subplot(2, 2, 2)
sns.histplot(data=iris_df, x='sepal_width', kde=True)
plt.title('Distribution of Sepal Width')

plt.subplot(2, 2, 3)
sns.histplot(data=iris_df, x='petal_length', kde=True)
plt.title('Distribution of Petal Length')

plt.subplot(2, 2, 4)
sns.histplot(data=iris_df, x='petal_width', kde=True)
plt.title('Distribution of Petal Width')

plt.tight_layout()
plt.show()

# Handling missing values by mean imputation
iris_df_mean_imputed = iris_df.fillna(iris_df.mean())

# Handling outliers by clipping or winsorization
iris_df_mean_imputed['petal_width'] = iris_df_mean_imputed['petal_width'].clip(upper=iris_df_mean_imputed['petal_width'].quantile(0.95))


plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(data=iris_df_mean_imputed, x='sepal_length', kde=True)
plt.title('Distribution of Sepal Length (Mean Imputation)')

plt.subplot(2, 2, 2)
sns.histplot(data=iris_df_mean_imputed, x='sepal_width', kde=True)
plt.title('Distribution of Sepal Width (Mean Imputation)')

plt.subplot(2, 2, 3)
sns.histplot(data=iris_df_mean_imputed, x='petal_length', kde=True)
plt.title('Distribution of Petal Length (Mean Imputation)')

plt.subplot(2, 2, 4)
sns.histplot(data=iris_df_mean_imputed, x='petal_width', kde=True)
plt.title('Distribution of Petal Width (Mean Imputation)')

plt.tight_layout()
plt.show()