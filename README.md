# 🛡️ Cyberbullying Detector

> An NLP-based text classification system that automatically detects cyberbullying across social media platforms using traditional machine learning and deep learning models.

---

## 📌 Overview

Cyberbullying is a growing crisis on platforms like Instagram, Twitter, TikTok, and WhatsApp — affecting millions of users, especially teenagers. Manual content moderation is slow, costly, and harmful to human reviewers. This project builds an automated pipeline that classifies social media text as **Cyberbullying** or **Normal** using three progressively powerful models: Logistic Regression, LSTM, and fine-tuned BERT.

---

## 🎯 Problem Statement

> *"Social media platforms lack the capability to automatically detect harmful text, leaving millions of users vulnerable to online harassment daily."*

---

## ✨ Key Features

- Scrapes raw, unlabeled data from Twitter, Reddit, and YouTube using public APIs
- Manual annotation pipeline with inter-annotator agreement
- Full preprocessing suite (lowercasing, emoji handling, tokenization, stopword removal)
- Three model comparison: Logistic Regression (baseline) → LSTM → BERT
- Evaluated on Accuracy, Precision, Recall, and F1-Score
- Outputs a prediction label with a confidence score
- Flags high-confidence cyberbullying detections with an alert label

---

## 🗂️ Project Structure

```
cyberbullying-detector/
│
├── data/
│   ├── raw/                  # Raw scraped data from APIs
│   ├── annotated/            # Manually labeled dataset
│   └── processed/            # Cleaned and tokenized data
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_annotation_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_logistic_regression.ipynb
│   ├── 05_lstm.ipynb
│   └── 06_bert_finetuning.ipynb
│
├── src/
│   ├── scraper/              # Twitter, Reddit, YouTube API scrapers
│   ├── preprocessing/        # Text cleaning and tokenization
│   ├── models/               # Model definitions and training scripts
│   └── evaluate/             # Evaluation metrics and comparison
│
├── results/
│   └── model_comparison.csv  # Accuracy, F1, Precision, Recall per model
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

Data is collected from scratch using public APIs — no pre-labeled datasets are used.

| Source     | Method                          | Est. Samples |
|------------|---------------------------------|--------------|
| Twitter/X  | Twitter API v2 (keyword search) | 400–500      |
| Reddit     | PRAW API (r/teenagers, r/roastme) | 300–400    |
| YouTube    | YouTube Data API v3 (comments)  | 200–300      |

**Total expected:** ~900–1,200 labeled samples

### Labels

| Label | Class | Description |
|-------|-------|-------------|
| `0`   | Normal | Safe, neutral, or positive content |
| `1`   | Cyberbullying | Harmful, offensive, or threatening content |

Annotation is performed by at least two team members. Disagreements are resolved by majority vote.

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+, pip

```bash
# Clone the repository
git clone https://github.com/your-username/cyberbullying-detector.git
cd cyberbullying-detector

# Install dependencies
pip install -r requirements.txt
```

### API Keys Required

Create a `.env` file in the root directory with your credentials:

```env
TWITTER_BEARER_TOKEN=your_token_here
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
YOUTUBE_API_KEY=your_api_key
```

---

## 🚀 Usage

### 1. Data Collection

```bash
python src/scraper/twitter_scraper.py
python src/scraper/reddit_scraper.py
python src/scraper/youtube_scraper.py
```

### 2. Preprocessing

```bash
python src/preprocessing/clean_text.py --input data/annotated/ --output data/processed/
```

### 3. Train Models

```bash
# Logistic Regression (baseline)
python src/models/train_logreg.py

# LSTM
python src/models/train_lstm.py

# BERT (fine-tuning)
python src/models/train_bert.py
```

### 4. Evaluate

```bash
python src/evaluate/compare_models.py
```

---

## 🔬 Methodology

### Pipeline

```
Data Collection → Annotation → Preprocessing → Feature Extraction → Model Training → Evaluation → Output
```

### Preprocessing Steps

- Lowercasing all text
- Removing URLs, hashtags, and special characters
- Emoji handling (removal or conversion to text)
- Tokenization via NLTK or HuggingFace tokenizer
- Optional stopword removal (disabled for BERT)
- Train / Validation / Test split: **70% / 15% / 15%**

### Models

| Model | Type | Features | Notes |
|-------|------|----------|-------|
| Logistic Regression | Traditional ML | TF-IDF | Baseline — fast and interpretable |
| LSTM | Deep Learning | Word Embeddings | Captures word order and sequence context |
| BERT | Transformer | Contextual Embeddings | Best performance — handles sarcasm and subtle language |

---

## 📊 Evaluation Metrics

| Metric | Why It Matters |
|--------|----------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of all flagged comments, how many are truly harmful |
| **Recall** | Of all harmful comments, how many were caught |
| **F1-Score** | Harmonic mean of Precision and Recall — primary metric |

---

## 📚 Related Work

| Paper | Key Contribution |
|-------|-----------------|
| Zhao et al. (2021) | Fine-tuned BERT on Twitter; outperforms SVM and Naive Bayes |
| Dadvar & Eckert (2022) | Cross-platform detection; highlights platform-specific language gaps |
| Alkomah & Ma (2022) | Survey of 60+ papers; LSTM and BERT consistently top performers |
| Huang et al. (2023) | RoBERTa + attention for sarcasm and indirect harassment detection |

---

## 🧩 Applications

- **Social media platforms** — auto-flag and remove harmful comments before they reach the victim
- **Schools & parents** — monitor online activity to protect children
- **Mental health platforms** — detect distress signals and alert support teams
- **Corporate environments** — detect workplace harassment over internal messaging tools

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Ensure that annotation guidelines are followed when contributing new labeled data.

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 👥 Team

Developed as part of an Artificial Intelligence — Text Classification course project.
