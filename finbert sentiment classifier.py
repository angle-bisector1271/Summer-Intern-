from transformers import pipeline

# Load FinBERT tone model
finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone", device=0)

# Input text
text = "profits at a 5 year low"

# Run prediction
result = finbert(text)[0]  # single prediction

# Normalize label (some versions return 'LABEL_0', others return 'Positive')
raw_label = result['label'].lower()

# Map any possible label version to sentiment
label_map = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive"
}

# Final output
output = {
    "label": label_map[raw_label],
    "score": round(result['score'], 4)
}

print(output)
