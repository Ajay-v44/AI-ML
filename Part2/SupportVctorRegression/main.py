import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

df=pd.read_csv("Position_Salaries.csv")

x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

print(x)
print(y)

y=y.reshape(len(y),1)
print(y)

sc_X=StandardScaler()
sc_y=StandardScaler()

x=sc_X.fit_transform(x)
y=sc_y.fit_transform(y)

print(x)
print(y)


regressor=SVR(kernel='rbf')
regressor.fit(x,y)

# print( sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))).reshape(-1,1))

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))


plt.scatter(sc_X.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


X_grid = np.arange(min(sc_X.inverse_transform(x)), max(sc_X.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()