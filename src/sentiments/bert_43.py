# for price classifier, sentiments by day using voting by most common and most confident(highest socore average)

from transformers import pipeline

# Load the pre-trained model and tokenizer
model = 'borisn70/bert-43-multilabel-emotion-detection'
tokenizer = 'borisn70/bert-43-multilabel-emotion-detection'

# Create a pipeline for sentiment analysis
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Test the model with a sentence
result = nlp("I feel great about this!")

# Print the result
print(result)
