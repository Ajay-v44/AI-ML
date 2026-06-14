import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import chain
from nltk import NaiveBayesClassifier
import matplotlib.pyplot as plt

data=pd.read_csv("book_reviews_sample.csv")
print(data.head())
print(data['reviewText'][0])

# lowercasing

data['reviewText_clean']=data['reviewText'].str.lower()
print(data['reviewText_clean'][0])

# remove punctuation
data['reviewText_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['reviewText_clean']), axis=1)
print(data['reviewText_clean'][0])

# vedaer sentiment analysis
vedar_sentiment=SentimentIntensityAnalyzer()

data['vedar_sentiment_core'] = data['reviewText_clean'].apply(lambda x: vedar_sentiment.polarity_scores(x))
print(data.head())

bins=[-1,-0.1,0.1,1]
labels=['negative','neutral','positive']
data['vedar_sentiment']=pd.cut(data['vedar_sentiment_core'].apply(lambda x: x['compound']),bins,labels=labels,include_lowest=True)
print(data.head())

data['vedar_sentiment'].value_counts().plot(kind='bar')
plt.show()

transformer_pipeline=pipeline('sentiment-analysis')
transformer_labels=[]

for review in data['reviewText_clean'].values:
    sentiment_list=transformer_pipeline(review)
    sentiment_label=[sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)
data['transformer_sentiment']=transformer_labels
print(data.head())
data['transformer_sentiment'].value_counts().plot(kind='bar')
plt.show()
