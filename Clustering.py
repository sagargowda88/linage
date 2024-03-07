import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file
data = pd.read_csv('your_file.csv')

# Preprocess the text data
text_data = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Convert all text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

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

# Visualize the clusters
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Add more colors if needed

for cluster_label in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    plt.scatter(tfidf_matrix_scaled[cluster_indices, 0], tfidf_matrix_scaled[cluster_indices, 1],
                c=colors[cluster_label], label=f'Cluster {cluster_label + 1}')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='black', marker='X', label='Centroids')
plt.title('Clustered Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Print one data point from each cluster
for cluster_label in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    sample_index = cluster_indices[0]
    print(f"Data point from Cluster {cluster_label + 1}:\n{data.iloc[sample_index]}")



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load the CSV file
data = pd.read_csv('your_file.csv')

# Preprocess the text data
text_data = data.apply(lambda x: ' '.join(x.astype(str)), axis=1)

# Convert all text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(text_data)

# Standardize the features
scaler = StandardScaler()
tfidf_matrix_scaled = scaler.fit_transform(tfidf_matrix.toarray())

# Dimensionality reduction with PCA
pca = PCA(n_components=2)
tfidf_matrix_pca = pca.fit_transform(tfidf_matrix_scaled)

# Elbow method to find the optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(tfidf_matrix_pca)
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
kmeans.fit(tfidf_matrix_pca)

# Visualize the clusters
cluster_labels = kmeans.labels_
colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Add more colors if needed

for cluster_label in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    plt.scatter(tfidf_matrix_pca[cluster_indices, 0], tfidf_matrix_pca[cluster_indices, 1],
                c=colors[cluster_label], label=f'Cluster {cluster_label + 1}')

plt.title('Clustered Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Print one data point from each cluster
for cluster_label in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    sample_index = cluster_indices[0]
    print(f"Data point from Cluster {cluster_label + 1}:\n{data.iloc[sample_index]}")

