import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

np.logical_xor(True, False)

np.logical_xor(True, True)

np.logical_xor(0, 0)

np.logical_xor(0, 1)

np.logical_xor([1,1,0,0],[1,0,1,0])

a = np.arange(5)
a

a < 2

np.logical_xor(a<1, a>3)


t = np.arange(10)
t

np.where(t%2)

np.where(t%2, 'Hi', 'Bye')

np.where(t<5, t, t**2)
# t = [0,1,2,3,4,5,6,7,8,9]
# np.where([1,1,1,1,1,0,0,0,0,0], 
#          [0,1,2,3,4,5,6,7,8,9],
#          [0,1,4,9,16,25,36,49,64,81])

np.where([1,0,1,0],
        [13,5,2,0],
        [9,-2,4,1])

np.where([[True, False], [True, True]],
        [[1,2],[3,4]],
        [[9,8],[7,6]])

# -1 is broadcast
a = np.array([[0,1,2],
             [0,2,4],
             [0,3,6]])
np.where(a<4, a, -1)

a

a<4

np.random.randn()

np.random.randn(3,2)

X = np.array([[1,2],[2,3],[10,19]])
y = np.array([0,1,0])

# np.where(y==0,X,'oops')

X[y==0]

X[:,1]

X[1,:]

X[1]

np.random.seed(0)
X = np.random.randn(200,2)
boolean = np.logical_xor(X[:,0]>0, X[:,1]>0)
y = np.where(boolean, 1, -1)

plt.scatter(X[y==1,0], 
           X[y==1, 1],
           c='blue', marker='x', label='1')

plt.scatter(X[y==-1,0], 
           X[y==-1, 1],
           c='red', marker='s', label='-1')

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.show()

def plot(classifier, C, gamma):
    x0 = np.linspace(-3,3,200)
    x1 = np.linspace(-3,3,200)
    
    x0_new, x1_new = np.meshgrid(x0,x1)
    
    x_t = np.c_[x0_new.ravel(), x1_new.ravel()]
    y_hat = classifier.predict(x_t).reshape(x0_new.shape)
    y_decision = classifier.decision_function(x_t).reshape(x0_new.shape)
    
    plt.contourf(x0_new, x1_new, y_decision,
                cmap=plt.cm.Greens, alpha=0.2)
    plt.contourf(x0_new, x1_new, y_hat,
                cmap=plt.cm.Blues, alpha=0.2)
    
    plt.scatter(X[:,0], X[:,1], c=y)
    score1 = classifier.score(X_test, y_test)
    score2 = classifier.score(X_train, y_train)
    
    plt.title(f'gamma = {gamma}, C={C} \n score train={score2} \n \
                score test = {score1}')
    plt.show()

X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.2)

### Kernel= 'rbf', gamma=0.1,1,10,100, C=1
### kernel='rbf',gamma=0.1, C=1,10,1000,10000,100000

def classifier(a=1, b=0.1):
    clf = SVC(kernel='rbf', C=a, gamma=b)
    clf.fit(X_train, y_train)
    plot(clf,a,b)

gamma = [0.1, 1, 10, 100]
C = [1, 10, 1000, 10000, 100000]

for i in gamma:
    classifier(b=i)

for i in C:
    classifier(a=i)
