import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X,_ = make_blobs(random_state=8, cluster_std=1)

plt.scatter(X[:,0],X[:,1])
plt.show()

clusters = AgglomerativeClustering(n_clusters = 2)

clusters.fit(X)

clusters.labels_

plt.scatter(X[:,0],X[:,1],c=clusters.labels_)
plt.show()

dendrogram

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method='single', metric='cosine'))

dendrogram = sch.dendrogram(sch.linkage(X, method='single', metric='euclidean'))

from sklearn.datasets import make_circles

X,_ = make_circles(200,noise=0.05, factor = 0.5)

plt.scatter(X[:,0],X[:,1])
plt.show()

clusters = AgglomerativeClustering(n_clusters = 2)

clusters.fit(X)

plt.scatter(X[:,0],X[:,1],c=clusters.labels_)
plt.show()

clusters = AgglomerativeClustering(n_clusters = 2, linkage= 'single')
clusters.fit(X)

plt.scatter(X[:,0],X[:,1],c=clusters.labels_)
plt.show()

dendrogram = sch.dendrogram(sch.linkage(X, method='single', metric='cosine'))
