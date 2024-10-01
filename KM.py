import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('Km.csv')

features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

kmeans.fit(features)

data['cluster'] = kmeans.predict(features)

centers = kmeans.cluster_centers_

plt.figure(figsize=(10, 8))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['cluster'], cmap='viridis', marker='o', edgecolor='k', s=100)

plt.scatter(centers[:, 1], centers[:, 2], c='red', marker='X', s=200, label='Centers') 
plt.title('K-Means Clustering of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
