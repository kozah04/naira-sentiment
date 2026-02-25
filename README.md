# Naira Sentiment Analysis

Sentiment analysis of Nigerian Pidgin (PCM) tweets on the Naira/CBN using the NaijaSenti dataset and a trained classification model demonstrated on manually collected tweets about the recent Naira appreciation against the USD.

---

## Project Structure

\```
naira-sentiment/
├── data/
│   ├── raw/          # NaijaSenti PCM train/dev/test splits
│   └── processed/    # Cleaned and encoded data
├── src/
│   ├── loader.py     # Data loading functions
│   ├── preprocess.py # Text cleaning and feature engineering
│   └── model.py      # Model training and evaluation
├── notebooks/
│   ├── exploration.ipynb  # EDA and data understanding
│   └── analysis.ipynb     # Model training, evaluation, demo
├── tests/
│   └── test_loader.py
├── demo/
│   └── tweets.csv    # Manually collected Naira tweets (gitignored)
└── outputs/          # Saved charts and results
\```

---

## Data

Training data from the [NaijaSenti](https://github.com/hausanlp/NaijaSenti) dataset — Nigerian Pidgin (PCM) split. Labels: positive, negative (neutral class dropped due to severe imbalance).

Demo tweets manually collected from Twitter/X, focused on public reaction to the Naira appreciation against the USD in 2025.

---

## Methods

- Text preprocessing: lowercasing, noise removal, tokenisation
- Features: TF-IDF
- Model: SVM (LinearSVC) with class weighting
- Evaluation: F1 score, confusion matrix, classification report
- Demo: model applied to manually collected Naira tweets

---

## Results

*To be completed after analysis.*

---

## Setup

\```bash
conda activate naira-sentiment
pip install -r requirements.txt
\```

---

## Author

Gwachat Kozah — [github.com/kozah04](https://github.com/kozah04)