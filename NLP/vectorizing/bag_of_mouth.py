import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

reviews=pd.read_csv("../book_reviews_sample.csv")
print(reviews.head())

reviews_list=reviews['reviewText'].values
print(reviews_list[:5])

# create the bag of words vectorizer
vectorizer=CountVectorizer()
countvec_fit=vectorizer.fit_transform(reviews_list)

bag_of_words=pd.DataFrame(countvec_fit.toarray(),columns=vectorizer.get_feature_names_out())
print(bag_of_words.head())