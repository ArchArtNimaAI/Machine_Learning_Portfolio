import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X,y = make_moons(n_samples=200, random_state=20, noise=0.1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

step1 = ('poly_features', PolynomialFeatures(degree=3))
step2 = ('SVM classifier', SVC(kernel='poly', degree=3, C=25))
# step3 = (,)
steps = [step1,step2]

clf = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf.fit(X_train, y_train)

x0 = np.linspace(-1.5,2.5,200)
x1 = np.linspace(-0.75,1.25,200)

x0_new, x1_new = np.meshgrid(x0,x1)

X_t = np.c_[x0_new.ravel(),x1_new.ravel()]
y_hat = clf.predict(X_t).reshape(x0_new.shape)
y_decision = clf.decision_function(X_t).reshape(x0_new.shape)

plt.contourf(x0_new, x1_new, y_decision, cmap=plt.cm.Greens, alpha=0.2)
plt.contourf(x0_new, x1_new, y_hat, cmap=plt.cm.Blues, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()
