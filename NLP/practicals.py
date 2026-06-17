import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (12,8)
default_plot_colour="#00bfbf"

data=pd.read_csv('./fake_news_data.csv')
print(data.head())

data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_colour)
plt.title('Count of Article Classification')
plt.ylabel('# of Articles')
plt.xlabel('Classification')
plt.show()


import seaborn as sns
import spacy
from spacy import displacy
from spacy import tokenizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


nlp = spacy.load('en_core_web_sm')
# split data by fake and factual news
fake_news = data[data['fake_or_factual'] == "Fake News"]
fact_news = data[data['fake_or_factual'] == "Factual News"]


# create spacey documents - use pipe for dataframe
fake_spaceydocs = list(nlp.pipe(fake_news['text']))
fact_spaceydocs = list(nlp.pipe(fact_news['text']))


# create function to extract tags for each document in our data
def extract_token_tags(doc:spacy.tokens.doc.Doc):
    return [(i.text, i.ent_type_, i.pos_) for i in doc]


# tag fake dataset 
fake_tagsdf = []
columns = ["token", "ner_tag", "pos_tag"]

for ix, doc in enumerate(fake_spaceydocs):
    tags = extract_token_tags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    fake_tagsdf.append(tags)
        
fake_tagsdf = pd.concat(fake_tagsdf)   

# tag factual dataset 
fact_tagsdf = []

for ix, doc in enumerate(fact_spaceydocs):
    tags = extract_token_tags(doc)
    tags = pd.DataFrame(tags)
    tags.columns = columns
    fact_tagsdf.append(tags)
        
fact_tagsdf = pd.concat(fact_tagsdf)  

print(fake_tagsdf.head())

columns = ['token', 'ner_tag', 'pos_tag']

print(fake_tagsdf[columns].head())

# token frequency count (fake)
pos_counts_fake = fake_tagsdf.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
pos_counts_fake.head(10)

# token frequency count (fact)
pos_counts_fact = fact_tagsdf.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
pos_counts_fact.head(10)



# frequencies of pos tags
pos_counts_fake.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10)
pos_counts_fact.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10)

print(pos_counts_fake[pos_counts_fake.pos_tag == "NOUN"][0:15])


# Named Entities


# top entities in fake news
top_entities_fake = fake_tagsdf[fake_tagsdf['ner_tag'] != ""] \
                    .groupby(['token','ner_tag']).size().reset_index(name='counts') \
                    .sort_values(by='counts', ascending=False)

# top entities in fact news
top_entities_fact = fact_tagsdf[fact_tagsdf['ner_tag'] != ""] \
                    .groupby(['token','ner_tag']).size().reset_index(name='counts') \
                    .sort_values(by='counts', ascending=False)

# create custom palette to ensure plots are consistent
ner_palette = {
    'ORG': sns.color_palette("Set2").as_hex()[0],
    'GPE': sns.color_palette("Set2").as_hex()[1],
    'NORP': sns.color_palette("Set2").as_hex()[2],
    'PERSON': sns.color_palette("Set2").as_hex()[3],
    'DATE': sns.color_palette("Set2").as_hex()[4],
    'CARDINAL': sns.color_palette("Set2").as_hex()[5],
    'PERCENT': sns.color_palette("Set2").as_hex()[6]
}

print(top_entities_fake)


sns.barplot(
    x = 'counts',
    y = 'token',
    hue = 'ner_tag',
    palette = ner_palette,
    data = top_entities_fake[0:10],
    orient = 'h',
    dodge=False
) \
.set(title='Most Common Entities in Fake News')
plt.show()

# Text Preprocessing
data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s*", "", x['text']), axis=1)
data['text_clean'] = data['text_clean'].str.lower()
data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

# stop words
en_stopwords = stopwords.words('english')
print(en_stopwords) # check this against our most frequent n-grams

data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# tokenize 
data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)

# lemmatize
lemmatizer = WordNetLemmatizer()
data["text_clean"] = data["text_clean"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# most common unigrams after preprocessing
tokens_clean = sum(data['text_clean'], [])
unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index()[:10]
print(unigrams)

unigrams['token'] = unigrams['index'].apply(lambda x: x[0]) # extract the token from the tuple so we can plot it

sns.barplot(x = "count", 
            y = "token", 
            data=unigrams,
            orient = 'h',
            palette=[default_plot_colour],
            hue = "token", legend = False)\
.set(title='Most Common Unigrams After Preprocessing')
plt.show()


# most common bigrams after preprocessing
bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()) 
print(bigrams[:10])

# Sentiment Analysis

# use vader so we also get a neutral sentiment count
vader_sentiment = SentimentIntensityAnalyzer()

data['vader_sentiment_score'] = data['text'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])


# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)


data['vader_sentiment_label'].value_counts().plot.bar(color=default_plot_colour)
plt.show()

sns.countplot(
    x = 'fake_or_factual',
    hue = 'vader_sentiment_label',
    palette = sns.color_palette("hls"),
    data = data
) \
.set(title='Sentiment by News Type')
plt.show()

# LDA


# fake news data vectorization
fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].reset_index(drop=True)
fake_news_text_str = fake_news_text.apply(lambda tokens: ' '.join(tokens))

countvec_lda = CountVectorizer()
doc_term_fake = countvec_lda.fit_transform(fake_news_text_str)

# generate perplexity scores to determine an optimum number of topics
perplexity_values = []
model_list = []

min_topics = 2
max_topics = 11

for num_topics_i in range(min_topics, max_topics+1):
    model = LatentDirichletAllocation(n_components=num_topics_i, random_state=0)
    model.fit(doc_term_fake)
    model_list.append(model)
    perplexity_values.append(model.perplexity(doc_term_fake))

plt.plot(range(min_topics, max_topics+1), perplexity_values)
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity score")
plt.legend(("perplexity_values"), loc='best')
plt.show()

# create lda model
num_topics_fake = 6

lda_model_fake = LatentDirichletAllocation(n_components=num_topics_fake, random_state=0)
lda_model_fake.fit(doc_term_fake)

feature_names_lda = countvec_lda.get_feature_names_out()
for topic_idx, topic in enumerate(lda_model_fake.components_):
    top_words = [feature_names_lda[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx}: {' '.join(top_words)}")


# TF-IDF & LSA

def tfidf_corpus(texts):
    tfidf = TfidfVectorizer()
    corpus_tfidf = tfidf.fit_transform(texts)
    return corpus_tfidf, tfidf

def get_coherence_scores(corpus, min_topics, max_topics):
    coherence_values = []
    model_list = []
    for num_topics_i in range(min_topics, max_topics+1):
        model = TruncatedSVD(n_components=num_topics_i, random_state=0)
        model.fit(corpus)
        model_list.append(model)
        coherence_values.append(model.explained_variance_ratio_.sum())
    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Explained Variance")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

corpus_tfidf_fake, tfidf_vectorizer = tfidf_corpus(fake_news_text_str)
get_coherence_scores(corpus_tfidf_fake, min_topics=2, max_topics=11)

# model for fake news data
lsa_fake = TruncatedSVD(n_components=5, random_state=0)
lsa_fake.fit(corpus_tfidf_fake)

feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lsa_fake.components_):
    top_words = [feature_names_tfidf[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic {topic_idx}: {' '.join(top_words)}")

print(data.head())

X = [','.join(map(str, l)) for l in data['text_clean']]
Y = data['fake_or_factual']

# text vectorization - CountVectorizer
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)


lr = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)


accuracy_score(y_pred_lr, y_test)


print(classification_report(y_test, y_pred_lr))

svm = SGDClassifier().fit(X_train, y_train)


y_pred_svm = svm.predict(X_test)


accuracy_score(y_pred_svm, y_test)
print(classification_report(y_test, y_pred_svm))