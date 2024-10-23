from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPRegressor, MLPClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape((-1,784))
X_test = x_test.reshape((-1,784))

mlp = MLPClassifier()
mlp.fit(X_train,y_train)

mlp.score(X_test, y_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,10,5,),max_iter=200)
mlp.fit(X_train,y_train)

mlp.score(X_test, y_test)
