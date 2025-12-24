# NLP-with-Disaster-Tweets

# Disaster Tweets Classification using BERT

## Kaggle Competition
**NLP with Disaster Tweets**  
https://www.kaggle.com/competitions/nlp-getting-started

## Kaggle Notebook
https://www.kaggle.com/code/saketjasuja/disaster-tweets-bert

---

## Problem Statement

The objective of this competition is to build a Natural Language Processing (NLP) model that can accurately classify whether a tweet refers to a real disaster or not.

Each tweet is labeled with:
- `1` → Tweet is about a real disaster  
- `0` → Tweet is not about a real disaster  

This is a binary text classification task and serves as an introductory NLP competition on Kaggle.

---

## Dataset Description

The dataset consists of tweets along with optional metadata.

### Files
- `train.csv` – Training data with labels
- `test.csv` – Test data without labels
- `sample_submission.csv` – Submission format

### Columns

| Column     | Description |
|------------|------------|
| `id`       | Unique tweet identifier |
| `text`     | Tweet content |
| `keyword`  | Keyword related to the tweet (may be missing) |
| `location` | Location of the tweet (may be missing) |
| `target`   | Binary label (1 = disaster, 0 = not disaster) |

---

## Approach

This solution uses **BERT (Bidirectional Encoder Representations from Transformers)** for tweet classification.

### Steps Followed

1. **Preprocessing**
   - Clean tweet text
   - Handle URLs, special characters, and noise
   - Tokenize text using BERT tokenizer

2. **Model**
   - Pretrained BERT model
   - Classification head added on top
   - Fine-tuned on disaster tweet data

3. **Training**
   - Optimizer: AdamW
   - Loss Function: CrossEntropyLoss
   - Validation split for performance monitoring

4. **Prediction**
   - Generate predictions for test data
   - Create `submission.csv` in Kaggle submission format

---

## Results

- Transformer-based model performs significantly better than traditional ML approaches
- Suitable for short and noisy text such as tweets
- Final evaluation and leaderboard score can be found in the Kaggle notebook

---

## How to Run

### On Kaggle
1. Open the notebook
2. Join the competition
3. Run all cells sequentially
4. Submit the generated `submission.csv`

### Local Setup (Optional)
```bash
pip install transformers torch scikit-learn pandas numpy
