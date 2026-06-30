"""
Document Classifier — Fine-tuned DistilBERT on AG News
ScholarX AI/ML Internship — Task 2

Classifies news articles into 4 categories: World, Sports, Business, Sci/Tech

Run this in Google Colab (free GPU: Runtime > Change runtime type > GPU)
"""

!pip install -q transformers datasets evaluate accelerate scikit-learn

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

# ----------------------------
# 1. Load dataset
# ----------------------------
dataset = load_dataset("ag_news")
label_names = dataset["train"].features["label"].names
print("Classes:", label_names)  # ['World', 'Sports', 'Business', 'Sci/Tech']

# Use a subset for faster training (remove .select() lines for full dataset)
train_data = dataset["train"].shuffle(seed=42).select(range(8000))
test_data = dataset["test"].shuffle(seed=42).select(range(2000))

# ----------------------------
# 2. Tokenize
# ----------------------------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=128)

train_tok = train_data.map(tokenize_fn, batched=True)
test_tok = test_data.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# 3. Load model
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label_names)
)

# ----------------------------
# 4. Metrics
# ----------------------------
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ----------------------------
# 5. Train
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("Final evaluation:", results)

# ----------------------------
# 6. Save model
# ----------------------------
trainer.save_model("document-classifier-model")
tokenizer.save_pretrained("document-classifier-model")
print("Model saved to ./document-classifier-model")
