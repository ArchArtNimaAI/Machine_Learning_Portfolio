from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

X,y = make_blobs(random_state=26, centers=3)

d = {0:'red', 1:'black', 2:'yellow'}
color = [d[i] for i in y]

plt.scatter(X[:,0], X[:,1], c=color)
plt.show()

#### Build the classifier

clf = KNeighborsClassifier()

# train the model
clf.fit(X,y)

x_min, x_max = X[:,0].min(), X[:,0].max()
y_min, y_max = X[:,1].min(), X[:,1].max()

x_test = np.linspace(x_min,x_max,300)
y_test = np.linspace(y_min,y_max,300)

xx,yy = np.meshgrid(x_test, y_test)
X_test = np.c_[xx.ravel(), yy.ravel()]
pred = clf.predict(X_test).reshape(xx.shape)

plt.contourf(xx,yy,pred, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=color)
plt.show()

#### new test

X,y = make_blobs(n_samples = 10000, n_features=20,
                centers=20, cluster_std=10, random_state=26)

plt.scatter(X[:,5], X[:,3], c=y)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=100, p=3, leaf_size = 500, weights='distance')

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

#### K-Fold crossvalidation

from sklearn.model_selection import cross_val_score

clf = KNeighborsClassifier(n_neighbors=100, p=3, leaf_size = 500, weights='distance')

# test_split & fit & score for k=5
cv_scores = cross_val_score(clf,X,y,cv=5)

cv_scores

np.mean(cv_scores)

### Iris dataset

from sklearn.datasets import load_iris

data = load_iris()

data.keys()

X = data.data

y = data.target

X.shape

clf = KNeighborsClassifier(n_neighbors=12, p=3, leaf_size = 50, weights='distance')

cv_scores = cross_val_score(clf,X,y,cv=5)

cv_scores

np.mean(cv_scores)
