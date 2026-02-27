# Naira Sentiment Analysis

Sentiment analysis of Nigerian Twitter discourse on the Naira/USD exchange rate using a fine-tuned AfriBERTa model trained on the NaijaSenti PCM dataset, demonstrated on 52 manually collected tweets from early 2026.

---

## Project Structure
```
naira-sentiment/
├── data/
│   ├── raw/          # NaijaSenti PCM train/dev/test splits
│   └── processed/    # Cleaned and encoded data
├── src/
│   ├── loader.py     # Data loading functions
│   ├── preprocess.py # Text cleaning and feature engineering
│   ├── model.py      # Model loading and inference
│   └── clean_demo.py # Demo tweet cleaning utility
├── notebooks/
│   ├── exploration.ipynb  # EDA on NaijaSenti PCM dataset
│   └── analysis.ipynb     # Sentiment analysis on Naira tweets
├── tests/
│   ├── test_loader.py
│   └── test_preprocess.py
├── demo/
│   └── tweets.xlsx   # Manually collected Naira tweets (gitignored)
└── outputs/          # Saved charts and results
```

---

## Dataset

Training data from the [NaijaSenti](https://github.com/hausanlp/NaijaSenti) dataset, Nigerian Pidgin (PCM) split.

| Split | Total | Negative | Positive |
|-------|-------|---------|---------|
| Train | 5,049 | 3,241 | 1,808 |
| Dev | 1,260 | 809 | 451 |
| Test | 3,723 | 2,326 | 1,397 |

Neutral class dropped due to severe underrepresentation (72 out of 5,122 training examples).

---

## Model

AfriBERTa (`castorini/afriberta_large`) fine-tuned for binary sentiment classification on NaijaSenti PCM tweets. Class weights applied during training to address the 64/36 negative/positive imbalance.

Trained model hosted on HuggingFace: [kozah04/naira-sentiment-afriberta](https://huggingface.co/kozah04/naira-sentiment-afriberta)

### Training Results

| Metric | Score |
|--------|-------|
| F1 Macro | 0.74 |
| F1 Weighted | 0.76 |
| Accuracy | 0.76 |
| Negative F1 | 0.81 |
| Positive F1 | 0.67 |

---

## Key Findings

- The AfriBERTa model performs well on casual pidgin sentiment but fails on formal economic commentary due to domain shift. This is a core finding of the project.
- For the demo analysis, Cardiff NLP's `twitter-roberta-base-sentiment-latest` was selected as a more appropriate model for the collected tweets.
- Nigerian public sentiment around the Naira appreciation is predominantly negative or skeptical (54%), despite objectively positive exchange rate movement.
- A significant portion of discourse (40%) is analytical and neutral, reflecting the educational nature of financial Twitter in Nigeria.
- Positive sentiment is rare (6%), limited to praise of specific actors like the CBN governor or personal expressions of hope.

---

## Setup
```bash
conda activate naira-sentiment
pip install -r requirements.txt
```

### Run Tests
```bash
python -m pytest tests/test_loader.py tests/test_preprocess.py -v
```

---

## Author

Gwachat Kozah — [github.com/kozah04](https://github.com/kozah04)