import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset = load_digits()
dataset

dataset.keys()

X = dataset.data
X.shape

images = dataset.images
images.shape

plt.imshow(images[300],cmap=plt.cm.gray)
plt.show()

y = dataset.target

y[300]

X[300]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

clf = SVC(kernel='linear', decision_function_shape='ovr', probability=True)

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

y_hat = clf.predict(X_test)[:10]
y_hat

y_test[:10]

for i,j,k in zip(X_test[:10],y_test,y_hat):
  t = i.reshape(8,8)
  plt.imshow(t,plt.cm.gray)
  plt.show()
  print ('\033[1m' + f"predicted class:{k}, ground truth = {j} ")
  print ("-"*10)


y_hat = clf.predict(X_test)
y_hat_prob = clf.predict_proba(X_test)

for i,j,k,l,m in zip(X_test,y_test,y_hat,range(len(X_test)),y_hat_prob):
  if j!=k:
    print('\033[1m' + f"test data number {l}")
    t = i.reshape(8,8)
    plt.imshow(t,plt.cm.gray)
    plt.show()
    plt.plot(range(10),m)
    plt.show()
    print (f"predicted class:{k}, predicted class probability = {np.where(m==np.amax(m))[0]} ground truth = {j} ")
    print ("-"*10)

for i,j,k,l,m in zip(X_test,y_test,y_hat,range(len(X_test)),y_hat_prob):
  p = np.where(m==np.amax(m))[0][0]
  if j!=p:
    print('\033[1m' + f"test data number {l}")
    t = i.reshape(8,8)
    plt.imshow(t,plt.cm.gray)
    plt.show()
    plt.plot(range(10),m)
    plt.show()
    print (f"predicted class:{k}, predicted class probability = {p} ground truth = {j} ")
    print ("-"*10)

h = clf.predict_proba(X_test[285:286])[0]
h

plt.plot(range(10),h)
plt.show()

clf.predict(X_test[285:286])[0]

np.amax(h)

np.where(h==np.amax(h))

**classification report**

print (classification_report(y_test,y_hat))

t = np.array([1,10,2,-4,12,5,3])
np.argmax(t)

y_hat_probability = np.array([])
for i in y_hat_prob:
  y_hat_probability = np.append(y_hat_probability, [np.argmax(i)])
y_hat_probability

y_hat_probability.shape

y_test.shape

print (classification_report(y_test,y_hat_probability))

print (confusion_matrix(y_test, y_hat))

print (confusion_matrix(y_test, y_hat_probability))
