# Document Classifier (DistilBERT)

A fine-tuned **DistilBERT** model that classifies news articles into 4 categories: **World, Sports, Business, Sci/Tech**.

Built as part of the **ScholarX AI/ML Internship — Task 2**.

## Overview

- **Base model:** `distilbert-base-uncased`
- **Dataset:** [AG News](https://huggingface.co/datasets/ag_news) (Hugging Face Datasets)
- **Framework:** Hugging Face `transformers` + `Trainer` API
- **Task:** Multi-class text classification (4 classes)

## Results

| Metric | Score |
|---|---|
| Accuracy | ~0.93 |
| Weighted F1 | ~0.93 |

*(Update with your actual numbers after training — printed at the end of `train.py`)*

## How it works

1. Loads the AG News dataset (120k news headlines + descriptions, 4 balanced classes)
2. Tokenizes text using the DistilBERT tokenizer (max length 128)
3. Fine-tunes DistilBERT for sequence classification over 2 epochs
4. Evaluates with accuracy and weighted F1 score
5. Saves the trained model for inference

## Usage

### 1. Train the model
Open `train.py` in **Google Colab** (enable GPU: Runtime → Change runtime type → GPU) and run all cells. This installs dependencies, trains, evaluates, and saves the model to `document-classifier-model/`.

### 2. Run inference
```bash
python inference.py
```

Example output:
```
Text: Apple unveils new AI chip set to power next generation iPhones.
Predicted: Sci/Tech (confidence: 0.97)
```

## Project structure
```
document-classifier/
├── train.py          # Training script (Colab-ready)
├── inference.py       # Load saved model and run predictions
└── README.md
```

## Tech stack

- Python
- Hugging Face Transformers & Datasets
- PyTorch
- scikit-learn / evaluate (metrics)

## Author

Namani — AI/ML Intern, ScholarX
