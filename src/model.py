import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "kozah04/naira-sentiment-afriberta"
DEMO_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def load_model():
    """
    Load the fine-tuned AfriBERTa model and tokenizer from HuggingFace Hub.
    Returns the model and tokenizer ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def predict(texts, model, tokenizer):
    """
    Run sentiment inference on a list of tweet strings.
    Returns a list of dicts with keys: tweet, sentiment, confidence.
    """
    label_map = {0: 'negative', 1: 'positive'}
    results = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()

        results.append({
            'tweet': text,
            'sentiment': label_map[pred_class],
            'confidence': round(confidence, 4)
        })

    return results

def load_demo_model():
    """
    Load Cardiff NLP's RoBERTa model fine-tuned on English tweets.
    Better suited for formal English economic commentary than AfriBERTa.
    """
    tokenizer = AutoTokenizer.from_pretrained(DEMO_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(DEMO_MODEL_NAME)
    model.eval()
    return model, tokenizer


def predict_demo(texts, model, tokenizer):
    """
    Run sentiment inference using Cardiff RoBERTa.
    Returns list of dicts with tweet, sentiment, confidence.
    Cardiff model uses labels: negative, neutral, positive.
    """
    results = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_class].item()

        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

        results.append({
            'tweet': text,
            'sentiment': label_map[pred_class],
            'confidence': round(confidence, 4)
        })

    return results