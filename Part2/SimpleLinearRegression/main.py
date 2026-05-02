import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# simple linear regression model
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)
y_pred=regressor.predict(X_train)
print(y_pred)
# the x train is the year of experince in  slaray csv data set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred, color = 'blue')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


print(regressor.predict([[1.1]]))
print(regressor.predict([[10.5]]))
