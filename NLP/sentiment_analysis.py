import transformers
from transformers import pipeline
sentiment_pipeline = pipeline('sentiment-analysis')
print(sentiment_pipeline("I love deep learning and AI!"))
print(sentiment_pipeline("i hate you"))