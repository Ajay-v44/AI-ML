import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import re
import pandas as pd

data=pd.read_csv('tripadvisor_hotel_reviews.csv')
data.info()