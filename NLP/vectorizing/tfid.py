import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

reviews=pd.read_csv("../book_reviews_sample.csv")
print(reviews.head())

reviews_list=reviews['reviewText'].values
print(reviews_list[:5])

# create the tf-idf vectorizer
tfidfvectorizer=TfidfVectorizer()
tfidf_fit=tfidfvectorizer.fit_transform(reviews_list)

tfidf_df=pd.DataFrame(tfidf_fit.toarray(),columns=tfidfvectorizer.get_feature_names_out())
print(tfidf_df.head())