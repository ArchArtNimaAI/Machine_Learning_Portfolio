from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.model_selection import train_test_split

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = DecisionTreeClassifier()

x_train.shape

X_train = x_train.reshape(-1,784)

X_train.shape

X_test = x_test.reshape(-1,784)

X_test.shape

y_train.shape

clf.fit(X_train, y_train)

clf.score(X_test, y_test)
