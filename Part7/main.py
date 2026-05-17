from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

dataset=pd.read_csv("./Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

nltk.download('stopwords')

# Stop words are the words that are not important for the analysis
# For eg: the, is, are, was, were, etc.
# Stemming is the process of removing the root word from the word
# For eg: loving -> love, loving -> loved, etc.

ps=PorterStemmer()
corpus=[]
for i in range(len(dataset)):
    review=re.sub('[^a-zA-Z]','',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review=' '.join(review)
    corpus.append(review)
print(corpus)

cv=CountVectorizer()

X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

print(X[0])
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# predict for x_test
y_pred=classifier.predict(X_test)

# print accuracy score
from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

# print classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))