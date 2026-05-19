# 🛡️ Cyberbullying Detector

An NLP-based text classification system that automatically detects cyberbullying across social media platforms using traditional machine learning and deep learning models — built completely from scratch as part of an AI course project.

---

## 📌 Overview

Cyberbullying is a growing crisis on platforms like Instagram, Twitter, TikTok, Reddit, and YouTube — affecting millions of users, especially teenagers. Manual content moderation is slow, costly, and harmful to human reviewers. This project builds an automated pipeline that classifies social media text as *Cyberbullying* or *Normal* using three progressively powerful models: Logistic Regression, LSTM, and fine-tuned BERT.

> *Problem Statement:* "Social media platforms lack the capability to automatically detect harmful text, leaving millions of users vulnerable to online harassment daily."

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression (Baseline) | 52.78% | 51.43% | 100.00% | 67.92% |
| *LSTM (Best)* | *91.67%* | *92.86%* | *90.28%* | *91.55%* |
| BERT (Transformer) | 84.72% | 89.06% | 79.17% | 83.82% |

*LSTM achieved the best overall performance* with 91.67% accuracy and 91.55% F1-Score.

---

## ✨ Key Features

- Scrapes real, unlabeled data from Reddit and YouTube using public APIs
- Manual annotation pipeline — labeled as Cyberbullying (1) or Normal (0)
- Full preprocessing suite built from scratch: lowercasing, URL/hashtag removal, custom tokenizer, manual stopword removal
- Three model comparison: Logistic Regression → LSTM → BERT
- All models implemented from scratch — no sklearn classifiers, no nn.LSTM, no BertForSequenceClassification
- Evaluated on Accuracy, Precision, Recall, and F1-Score — all computed from TP/FP/TN/FN without sklearn.metrics
- Balanced dataset with 70/15/15 train/validation/test split and zero overlap between splits

---

## 🗂️ Project Structure


Cyberbullying-Detector/
│
├── data_collection/
│   ├── collect_reddit.py         # Reddit public JSON API scraper
│   ├── collect_youtube.py        # YouTube Data API v3 scraper
│   ├── merge.py                  # Merge Reddit + YouTube datasets
│   ├── annotate.py               # Keyword-based auto-labeling
│   ├── cleanup.py                # Remove junk rows and noise
│   ├── rebuild_dataset.py        # Augment + balance dataset
│   ├── fix_splits.py             # Fix overlaps and rebalance splits
│   └── drop_source.py            # Final column cleanup
│
├── preprocessing/
│   └── preprocess.py             # Full preprocessing pipeline from scratch
│
├── models/
│   ├── tfidf.py                  # TF-IDF vectorizer from scratch
│   ├── logistic_regression.py    # Logistic Regression from scratch
│   ├── train_lr.py               # Train LR model
│   ├── lstm_model.py             # LSTM with manual gate matrices (nn.Parameter)
│   ├── train_lstm.py             # Train LSTM model
│   ├── bert_model.py             # BERT + custom classification head
│   ├── train_bert.py             # Fine-tune BERT (custom loop, no Trainer)
│   ├── evaluate.py               # Metrics from scratch (Acc/Prec/Recall/F1)
│   └── compare_models.py         # Final model comparison table
│
├── dataset/
│   ├── train.csv                 # 768 samples (384 each class)
│   ├── val.csv                   # 138 samples (69 each class)
│   └── test.csv                  # 144 samples (72 each class)
│
├── saved_models/
│   ├── lr_model.pkl
│   ├── lstm_model.pt
│   ├── bert_model.pt
│   └── final_comparison.csv
│
└── README.md


---

## 📦 Dataset

Data collected from scratch using public APIs — no pre-labeled Kaggle datasets used.

| Source | Method | Collected |
|--------|--------|-----------|
| Reddit | Public JSON API (no key needed) — r/teenagers, r/roastme, r/AITA, r/confession, r/unpopularopinion | 1,000 comments |
| YouTube | YouTube Data API v3 — 5 cyberbullying-related search queries | 715 comments |
| *Total after cleaning + balancing* | | *1,050 samples* |

### Labels

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Normal | Safe, neutral, or positive content |
| 1 | Cyberbullying | Harmful, offensive, or threatening content |

### Split

| Split | Samples | Class 0 | Class 1 |
|-------|---------|---------|---------|
| Train | 768 | 384 | 384 |
| Val | 138 | 69 | 69 |
| Test | 144 | 72 | 72 |

---

## ⚙️ Installation

*Prerequisites:* Python 3.8+, pip

bash
# Clone the repository
git clone https://github.com/Momina-13/Cyberbullying-Detector.git
cd Cyberbullying-Detector

# Install dependencies
pip install numpy pandas torch transformers requests


*For YouTube data collection only:*

YOUTUBE_API_KEY=your_key_here

Reddit collection requires no API key.

---

## 🚀 Usage

### 1. Data Collection (already done — dataset included)
bash
python data_collection/collect_reddit.py
python data_collection/collect_youtube.py
python data_collection/merge.py


### 2. Preprocessing
bash
python preprocessing/preprocess.py


### 3. Train Models
bash
# Logistic Regression (baseline)
python models/train_lr.py

# LSTM (from scratch)
python models/train_lstm.py

# BERT (fine-tuning with custom head)
python models/train_bert.py


### 4. Compare All Models
bash
python models/compare_models.py


---

## 🔬 Methodology

### Preprocessing Steps (all from scratch — no NLTK)
1. Lowercase all text
2. Remove URLs, hashtags, @mentions using regex
3. Remove special characters — keep only letters, numbers, spaces
4. Custom whitespace tokenizer (no library wrappers)
5. Manual stopword removal using a hardcoded list of 100+ words
6. 70/15/15 split with zero overlap and balanced classes

### Models

| Model | Type | Implementation | Key Constraint Met |
|-------|------|---------------|-------------------|
| Logistic Regression | Traditional ML | TF-IDF + sigmoid + gradient descent | No sklearn |
| LSTM | Deep Learning | All 4 gate matrices as nn.Parameter | No nn.LSTM |
| BERT | Transformer | BertModel + custom nn.Sequential head | No BertForSequenceClassification, no Trainer API |

### Evaluation Metrics (computed from scratch)

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + FP + TN + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × Precision × Recall / (Precision + Recall) |

---

## 📚 Related Work

| Paper | Key Contribution |
|-------|-----------------|
| Zhao et al. (2021) | Fine-tuned BERT on Twitter; outperforms SVM and Naive Bayes |
| Dadvar & Eckert (2022) | Cross-platform detection; highlights platform-specific language gaps |
| Alkomah & Ma (2022) | Survey of 60+ papers; LSTM and BERT consistently top performers |
| Huang et al. (2023) | RoBERTa + attention for sarcasm and indirect harassment detection |

---

## ⚠️ Limitations

- Annotation was keyword-based with manual review — some label noise is present
- Augmented cyberbullying samples (text variations) reduce linguistic diversity
- Dataset is English-only; a small number of non-English comments exist
- BERT fine-tuned for only 3 epochs due to compute constraints

---

## 👥 Team

| Member | Role |
|--------|------|
| Ezzah Noor | Data collection, preprocessing pipeline, presentation |
| Eiman Farooq | TF-IDF, Logistic Regression, LSTM — all from scratch |
| Momina Qayyum | BERT fine-tuning, evaluation metrics, model comparison |

*Course:* Artificial Intelligence — Text Classification Project
*Institution:* BSCS Section B | 2026
*Repository:* [github.com/Momina-13/Cyberbullying-Detector](https://github.com/Momina-13/Cyberbullying-Detector)
