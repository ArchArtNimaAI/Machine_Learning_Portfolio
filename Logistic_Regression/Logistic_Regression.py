from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

(x_train, y_train), (x_test, y_test) = mnist.load_data()

clf = LogisticRegression(n_jobs=-1)

x_test.shape

X_train = x_train.reshape((-1,784))
X_test = x_test.reshape((-1,784))

X_test.shape

clf.fit(X_train, y_train)

clf.predict(X_test[:10])

for i in X_test[:10]:
  plt.imshow(i.reshape((28,28)), cmap=plt.cm.gray_r)
  plt.show()

clf.score(X_test, y_test)

cm = metrics.confusion_matrix(y_test, clf.predict(X_test))
print (cm)
