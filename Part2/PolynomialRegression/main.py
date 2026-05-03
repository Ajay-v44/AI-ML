import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('./Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

lin_reg=LinearRegression()
lin_reg.fit(X_train,y_train)


# Polunomial Regression

poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
print(X_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


plt.scatter(X,y, color='red')
plt.scatter(X,lin_reg.predict(X), color='black')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X,y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()