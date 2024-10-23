from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape((-1,784))
X_test = x_test.reshape((-1,784))

X_train.shape

pca = PCA(225)

low_dimension = pca.fit_transform(X_train)

pca.n_components_

low_dimension.shape

approximation = pca.inverse_transform(low_dimension)

approximation.shape

plt.figure(figsize=(8,4))

#plot original
plt.subplot(1,3,1)
plt.imshow(x_train[0], cmap=plt.cm.gray_r)
plt.xlabel('original 784 feature')
plt.title('original')

#plot low dimension
plt.subplot(1,3,2)
plt.imshow(low_dimension[0].reshape(15,15), cmap=plt.cm.gray_r)
plt.xlabel('original 225 feature')
plt.title('low dimension')

#plot approximation
plt.subplot(1,3,3)
plt.imshow(approximation[0].reshape(28,28), cmap=plt.cm.gray_r)
plt.xlabel('approximation 784 feature')
plt.title('approximation')

plt.show()

pca.explained_variance_ratio_

np.sum(pca.explained_variance_ratio_)

pca.singular_values_
