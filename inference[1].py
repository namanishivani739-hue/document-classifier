"""
Inference demo for the fine-tuned document classifier.
Run after train.py has saved the model to ./document-classifier-model
"""

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="document-classifier-model",
    tokenizer="document-classifier-model",
)

label_map = {
    "LABEL_0": "World",
    "LABEL_1": "Sports",
    "LABEL_2": "Business",
    "LABEL_3": "Sci/Tech",
}

samples = [
    "Apple unveils new AI chip set to power next generation iPhones.",
    "The national team secured a dramatic victory in the final minutes of the match.",
    "Stock markets rallied today after the central bank's interest rate decision.",
    "Diplomats from several nations met to discuss the ongoing trade negotiations.",
]

for text in samples:
    pred = classifier(text)[0]
    label = label_map.get(pred["label"], pred["label"])
    print(f"Text: {text}\nPredicted: {label} (confidence: {pred['score']:.2f})\n")
