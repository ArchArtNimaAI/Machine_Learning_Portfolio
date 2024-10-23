import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

df = pd.read_csv("PLRdataset.csv")
df.head()

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

plt.scatter(X,y,color='blue')
plt.show()

X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.2)

#### test on simple linear regression

test1 = LinearRegression()
test1.fit(X_train, y_train)

test1.score(X_test, y_test)

plt.scatter(X,y,color='blue')
plt.plot(X, test1.predict(X), color='red')
plt.show()

#### build polynomial model

# step1: polynomial conversion of variables
poly = PolynomialFeatures(degree=2)
X_train_new = poly.fit_transform(X_train)
X_test_new = poly.fit_transform(X_test)

X_train[0]

X_train_new[0]

# step2: multiple linear regression
test2 = LinearRegression()
test2.fit(X_train_new, y_train)

test2.score(X_test_new, y_test)

X_train_sorted = []
X_train_new_sorted = []

t = sorted(zip(X_train, X_train_new), key=lambda h: h[0])

for i,j in t:
    X_train_sorted.append(i)
    X_train_new_sorted.append(j)

plt.scatter(X,y,color='blue')
plt.plot(X_train_sorted, test2.predict(X_train_new_sorted), color='red')
plt.show()

#### degree = 5

# step1: polynomial conversion of variables
poly = PolynomialFeatures(degree=5)
X_train_new = poly.fit_transform(X_train)
X_test_new = poly.fit_transform(X_test)

# step2: multiple linear regression
test3 = LinearRegression()
test3.fit(X_train_new, y_train)

test3.score(X_test_new, y_test)

X_train_sorted = []
X_train_new_sorted = []

t = sorted(zip(X_train, X_train_new), key=lambda h: h[0])

for i,j in t:
    X_train_sorted.append(i)
    X_train_new_sorted.append(j)

plt.scatter(X,y,color='blue')
plt.plot(X_train_sorted, test3.predict(X_train_new_sorted), color='red')
plt.show()

#### degree=10

# step1: polynomial conversion of variables
poly = PolynomialFeatures(degree=5)
X_train_new = poly.fit_transform(X_train)
X_test_new = poly.fit_transform(X_test)

# step2: multiple linear regression
test4 = LinearRegression()
test4.fit(X_train_new, y_train)

test4.score(X_test_new, y_test)

X_train_sorted = []
X_train_new_sorted = []

t = sorted(zip(X_train, X_train_new), key=lambda h: h[0])

for i,j in t:
    X_train_sorted.append(i)
    X_train_new_sorted.append(j)

plt.scatter(X,y,color='blue')
plt.plot(X_train_sorted, test4.predict(X_train_new_sorted), color='red')
plt.show()



plt.plot([2,2.5,3],[10,5,20])

g = [(1,2),(4,3),(2,6),(0,4),(-2,9)]
sorted(g,key=lambda j:j[1])
