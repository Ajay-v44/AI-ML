from transformers import pipeline,AutoTokenizer

sentiment_classifier=pipeline("sentiment-analysis")

print(sentiment_classifier("i am so excited"))

zeroshot_classifier=pipeline("zero-shot-classification",model="facebook/bart-large-mnli")
sequnece_to_classify="one day i will see the world"
candidate_labels=['travel','cooking','dancing']
print(zeroshot_classifier(sequnece_to_classify,candidate_labels))

model="bert-base-uncased"

tokenizer=AutoTokenizer.from_pretrained(model)

sentence="I'm so excited to be learning about large langugae models"

input_ids=tokenizer(sentence, return_tensors="pt")
print(input_ids)


from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch

print(sentence)
print(input_ids)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

with torch.no_grad():
    logits = model(**input_ids).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

model_directory = "./"

tokenizer.save_pretrained(model_directory)

model.save_pretrained(model_directory)


my_tokenizer = AutoTokenizer.from_pretrained(model_directory)

my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)
