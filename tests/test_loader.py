import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.loader import load_split, load_all

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))


class TestLoadSplit:

    def test_returns_dataframe(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert list(df.columns) == ['tweet', 'label']

    def test_no_neutral_labels(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert 'neutral' not in df['label'].values

    def test_only_valid_labels(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert set(df['label'].unique()).issubset({'positive', 'negative'})

    def test_index_is_reset(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert df.index.tolist() == list(range(len(df)))

    def test_no_empty_dataframe(self):
        df = load_split(os.path.join(DATA_DIR, 'train.tsv'))
        assert len(df) > 0


class TestLoadAll:

    def test_returns_three_dataframes(self):
        train, dev, test = load_all(DATA_DIR)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(dev, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_all_splits_have_correct_columns(self):
        train, dev, test = load_all(DATA_DIR)
        for df in [train, dev, test]:
            assert list(df.columns) == ['tweet', 'label']

    def test_no_neutral_in_any_split(self):
        train, dev, test = load_all(DATA_DIR)
        for df in [train, dev, test]:
            assert 'neutral' not in df['label'].values

    def test_train_is_largest_split(self):
        train, dev, test = load_all(DATA_DIR)
        assert len(train) > len(dev)