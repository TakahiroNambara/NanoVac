import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score

data = np.array([
    [1.0, 0.57, 0.59, 0.71],
    [1.0, 0.53, 0.65, 0.67],
    [1.0, 0.57, 0.76, 0.64],
    [0.72, 1.0, 0.70, 0.74],
    [0.62, 1.0, 0.44, 0.69],
    [0.52, 1.0, 0.79, 0.50],
    [0.59, 0.69, 1.0, 0.76],
    [0.61, 0.57, 1.0, 0.74],
    [0.34, 0.56, 1.0, 0.53],
    [0.57, 0.66, 0.61, 1.0],
    [0.75, 0.48, 0.55, 1.0],
    [0.74, 0.59, 0.75, 1.0],
], dtype=np.float64)

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
data_2d = tsne.fit_transform(data).astype(np.float64)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data_2d)
kmeans_labels = kmeans.labels_

manual_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

label_names = ['"1"', '"6"', '"5"', '"7"']
colors = ['red', 'green', 'blue', 'purple']

h = .5  
x_min, x_max = data_2d[:, 0].min() - 5, data_2d[:, 0].max() + 5
y_min, y_max = data_2d[:, 1].min() - 5, data_2d[:, 1].max() + 5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h, dtype=np.float64),
    np.arange(y_min, y_max, h, dtype=np.float64)
)

grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
Z = kmeans.predict(grid_points)
Z = Z.reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           origin='lower', cmap='Pastel2', alpha=1.0)

for i in range(4):
    idxs = np.where(manual_labels == i)
    plt.scatter(data_2d[idxs, 0], data_2d[idxs, 1], c=colors[i],
                label=f'digt {label_names[i]}', edgecolor='k')

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='black', zorder=10, label='K-means Centroids')

plt.title('k-means Clustering with Manual Labels on t-SNE Reduced Data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.grid(True)
plt.show()


silhouette = silhouette_score(data_2d, kmeans_labels)
print("Silhouette Score: ", silhouette)

ari = adjusted_rand_score(manual_labels, kmeans_labels)
print("Adjusted Rand Index: ", ari)

ami = adjusted_mutual_info_score(manual_labels, kmeans_labels)
print("Adjusted Mutual Information: ", ami)
