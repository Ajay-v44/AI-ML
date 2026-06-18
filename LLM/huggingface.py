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

input_ids=tokenizer(sentence)
print(input_ids)




