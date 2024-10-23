import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=200, centers=2, n_features=2,
                  random_state=30)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

X[0]

y[0]

len(y[y==1])

#### test split

X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.7)

#### Build the SVC model

clf = SVC(kernel='linear')
clf.fit(X_train,y_train)

clf.coef_

clf.intercept_

W = clf.coef_[0]
b = clf.intercept_[0]

clf.support_vectors_

def plot_svm(classifier, bounds):
    W = classifier.coef_[0]
    b = classifier.intercept_[0]
    
    # bounds = [-10,10]
    x0 = np.linspace(bounds[0], bounds[1], 100)
    # w1.x1 + w0.x0 + b = 0
    x1 = -(W[0]*x0 + b) / W[1]
    
    shift = 1 / W[1]
    margin_plane1 = x1 + shift
    margin_plane2 = x1 - shift
    
    support_vectors = classifier.support_vectors_
    
    plt.scatter(support_vectors[:,0], support_vectors[:,1], c='red', alpha=0.5, s=100)
    
    plt.plot(x0,x1, 'r-', linewidth=3)
    plt.plot(x0,margin_plane1, 'b--', linewidth=1)
    plt.plot(x0,margin_plane2, 'b--', linewidth=1)

plt.scatter(X[:,0], X[:,1], c=y)
fig_data = plt.gca()
x0_lim = fig_data.get_xlim()
plot_svm(clf,x0_lim)
plt.show()

# sample for prediction
sample = np.array([[0,-10], [6,0], [1.5,-5]])
clf.predict(sample)

clf.decision_function(sample)

clf.score(X_test,y_test)

for i,j in zip(y_test, clf.predict(X_test)):
    print (f"the real class is {i} and the predicted class is {j}")

# K-fold cross-validation
