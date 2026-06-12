import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
import pandas as pd

nltk.download('punkt_tab')
nltk.download('wordnet')

data=pd.read_csv('tripadvisor_hotel_reviews.csv')
print(data.info())

print(data.head())

print(data['Review'][0])

# lower casing
data['review_lower_case']=data['Review'].str.lower()

print(data.head())

# stop word

en_stopwords=stopwords.words('english')

en_stopwords.remove("not")

data['review_no_stopwords']=data['review_lower_case'].apply(lambda x: ' '.join ([word for word in word_tokenize(x) if word not in en_stopwords]))


data['review_no_stopwords_no_punct']=data.apply(lambda x : re.sub(r"[*]","star",x['review_no_stopwords']),axis=1)

print(data.head())

data['review_no_stopwords_no_punct']=data.apply(lambda x:re.sub(r"([^\w\s])"," ",x['review_no_stopwords_no_punct']),axis=1)

data['tokenized']=data.apply(lambda x:word_tokenize(x['review_no_stopwords_no_punct']),axis=1)

print(data['tokenized'][0],"tokenized")

ps=PorterStemmer()
data['stemmed']=data['tokenized'].apply(lambda tokens:[ps.stem(word) for word in tokens])

print(data['stemmed'])

lematizer=WordNetLemmatizer()

data['lemmatized']=data['tokenized'].apply(lambda tokens:[lematizer.lemmatize(token) for token in tokens])
print(data['lemmatized'][0])

token_clean=sum(data['lemmatized'],[])

unigrams=(pd.Series(nltk.ngrams(token_clean,1))).value_counts()
print(unigrams)

bigram=(pd.Series(nltk.ngrams(token_clean,2))).value_counts()
print(bigram)


