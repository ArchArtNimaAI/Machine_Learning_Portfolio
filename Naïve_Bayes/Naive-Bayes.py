from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X,y = make_blobs(random_state=10)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

clf = GaussianNB()

clf.fit(X,y)

x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()

x_test = np.linspace(x_min,x_max,300)
y_test = np.linspace(y_min,y_max,300)

xx,yy = np.meshgrid(x_test, y_test)
X_test = np.c_[xx.ravel(), yy.ravel()]
pred = clf.predict(X_test).reshape(xx.shape)

plt.contourf(xx,yy,pred, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

#### new test

X,y = make_blobs(n_samples = 10000, n_features=20,
                centers=20, cluster_std=10, random_state=26)

plt.scatter(X[:,5], X[:,3], c=y)
plt.show()

from sklearn.model_selection import cross_val_score

clf = GaussianNB()

cv_scores = cross_val_score(clf,X,y,cv=5)

cv_scores

np.mean(cv_scores)
