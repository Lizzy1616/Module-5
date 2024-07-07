# Module-5
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
columns = data.feature_names

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA components
df_pca = pd.DataFrame(data=X_pca, columns=['PCA Component 1', 'PCA Component 2'])

# Print the explained variance ratio
print("Explained Variance Ratio of the two components:")
print(pca.explained_variance_ratio_)

# Output the DataFrame with PCA components
print("\nDataFrame with PCA components:")
print(df_pca.head())
