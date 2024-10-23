from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

X,_ = make_blobs(random_state=20, centers=4, cluster_std=1)

plt.scatter(X[:,0], X[:,1])
plt.show()

cls = KMeans(n_clusters=3)

cls.fit(X)

wcss = cls.inertia_

wcss

wcss = []
for k in range(1,10):
  cls = KMeans(n_clusters=k)
  cls.fit(X)
  wcss.append(cls.inertia_)

wcss.insert(0,None)

wcss

plt.plot(wcss)
plt.xlabel('number of clusters(K)')
plt.ylabel('wcss')
plt.show()

cls = KMeans(n_clusters=2)
cls.fit(X)

x0 = np.linspace(-12,11,200)
x1 = np.linspace(-3,11,200)

x0_new, x1_new = np.meshgrid(x0,x1)

X_t = np.c_[x0_new.ravel(),x1_new.ravel()]
clusters = cls.predict(X_t).reshape(x0_new.shape)


plt.contourf(x0_new, x1_new, clusters, cmap=plt.cm.Greens)

plt.scatter(X[:,0], X[:,1])
plt.show()
