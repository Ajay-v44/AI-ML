import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values



labelencoder_X = LabelEncoder()
X[:,2]= labelencoder_X.fit_transform(X[:,2])

print(X)

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], 
    remainder='passthrough'
)
X = ct.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann=tf.keras.models.Sequential()


ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

ann.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ann.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=100
)

seolution=ann.predict(sc.transform([
    [1, 0, 1, 600, 1,40, 3, 60000,2,1,1,50000]
])>0.5)
print(seolution)

Y_pred = ann.predict(X_test)
Y_pred = (Y_pred > 0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)

