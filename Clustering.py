import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the CSV file
data = pd.read_csv('your_file.csv')

# Convert all text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data.values.astype('U'))

# Standardize the features
scaler = StandardScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix.toarray())

# Elbow method to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(tfidf_matrix_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# From the elbow curve, select the optimal number of clusters
optimal_k = 3  # Adjust this based on the elbow plot

# Perform k-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(tfidf_matrix_scaled)

# Visualize the clusters (Note: visualization might not be optimal for high-dimensional data)
# As an alternative, you can reduce the dimensionality using techniques like PCA or t-SNE
# and then visualize the clusters in lower dimensions.
# For simplicity, let's skip visualization for now.

# Print one data point from each cluster
for cluster in range(optimal_k):
    cluster_indices = np.where(kmeans.labels_ == cluster)[0]
    sample_index = cluster_indices[0]
    print(f"Data point from Cluster {cluster + 1}:\n{data.iloc[sample_index]}")
