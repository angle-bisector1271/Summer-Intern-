from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "inflation at an all time low, investors are gaining confidence"
labels = ["positive", "neutral", "negative"]
result = classifier(text, candidate_labels=labels)
print(result)
