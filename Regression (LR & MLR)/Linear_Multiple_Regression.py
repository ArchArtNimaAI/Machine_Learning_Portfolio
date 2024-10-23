import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# dataframe
df = pd.read_csv('AmesHousing_simple.csv')
df.head()
df.tail(7)

# df.iloc[row,column]
# X = df.iloc[:,[0]].values
X = df.iloc[:,:-1]
X_values = X.values
# X should have a vector structure
y = df.iloc[:,-1]
y_values = y.values

X_values.shape

y_values.shape

### Plot the Features vs y(Sale prices)
features_names = X.columns
plt.figure(figsize=(15,10))

for i, feature in enumerate(features_names):
    plt.subplot(3,3,i+1)
    plt.scatter(X[feature],y)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    
plt.tight_layout()
plt.show()

### split train and test dataset
X_train, X_test, y_train, y_test = \
train_test_split(X,y,test_size=0.2, random_state=42)


X_train.shape

X_test.shape

y_test.shape

y_train.shape

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

##  Create the Regression Model 

regressor = LinearRegression()

#### Train the model
# traing - learn
regressor.fit(X_train, y_train)

regressor.coef_

W = regressor.coef_

regressor.intercept_

b = regressor.intercept_

y_pred = regressor.predict(X_test)
###Scores###
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

# Mean Squared Error (MSE): 2367381219.3921275
# R-squared (R²): 0.704725030671056


#### Plot the model

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# y_pred
for i,j in zip(y_pred, y_test):
    print (f"prediction = {i}, real = {j} ")




#### Prediction results based on the first column(BsmtFin SF 1):
# Mean Squared Error (MSE): 6449201903.830517
# R-squared (R²): 0.19561417538038628

#### Prediction results based on the first column(Bsmt Unf SF):
# Mean Squared Error (MSE): 7723568807.397873
# R-squared (R²): 0.036666961774758255

#### Prediction results based on the first column(Total Bsmt SF	):
# Mean Squared Error (MSE): 4334447029.814333
# R-squared (R²): 0.4593799666472864

#### Prediction results based on the first column(1st Flr SF):
# Mean Squared Error (MSE): 4595924072.534869
# R-squared (R²): 0.4267668728468067

#### Prediction results based on the first column(2nd Flr SF):
# Mean Squared Error (MSE): 7674531090.837479
# R-squared (R²): 0.04278326029682289

#### Prediction results based on the first column(Gr Liv Area):
# Mean Squared Error (MSE): 3821184066.2726417
# R-squared (R²): 0.5233974153691151
