import pandas as pd
import os

def load_split(filepath):
    """
    Load a single NaijaSenti .tsv file and return a cleaned DataFrame.
    Drops neutral class, resets index, renames columns for clarity.
    """
    df = pd.read_csv(filepath, sep='\t')
    df.columns = ['tweet', 'label']
    df = df[df['label'] != 'neutral']
    df = df.reset_index(drop=True)
    return df


def load_all(data_dir):
    """
    Load train, dev, and test splits from the given directory.
    Returns three DataFrames: train, dev, test.
    """
    train = load_split(os.path.join(data_dir, 'train.tsv'))
    dev   = load_split(os.path.join(data_dir, 'dev.tsv'))
    test  = load_split(os.path.join(data_dir, 'test.tsv'))
    return train, dev, test