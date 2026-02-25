import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_text(text):
    """
    Clean a single tweet string.
    Lowercases, removes URLs, mentions, hashtags, punctuation, and extra whitespace.
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'@\w+', '', text)                   # remove mentions
    text = re.sub(r'#\w+', '', text)                   # remove hashtags
    text = re.sub(r'[^\w\s]', '', text)                # remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()           # remove extra whitespace
    return text


def preprocess(df):
    """
    Apply cleaning and label encoding to a DataFrame.
    Returns DataFrame with cleaned tweets and encoded labels (positive=1, negative=0).
    """
    df = df.copy()
    df['tweet'] = df['tweet'].apply(clean_text)
    df = df[df['tweet'].str.strip() != '']             # drop empty tweets after cleaning
    df = df.reset_index(drop=True)

    encoder = LabelEncoder()
    df['label_encoded'] = encoder.fit_transform(df['label'])

    return df, encoder