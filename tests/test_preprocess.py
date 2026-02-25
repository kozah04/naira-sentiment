import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import clean_text, preprocess


class TestCleanText:

    def test_lowercases_text(self):
        assert clean_text("NAIRA IS RISING") == "naira is rising"

    def test_removes_urls(self):
        assert clean_text("check this https://t.co/abc123") == "check this"

    def test_removes_mentions(self):
        assert clean_text("@CBNgov una do well") == "una do well"

    def test_removes_hashtags(self):
        assert clean_text("naira don rise #Naira") == "naira don rise"

    def test_removes_punctuation(self):
        assert clean_text("e don happen!!!") == "e don happen"

    def test_removes_extra_whitespace(self):
        assert clean_text("naira   don   rise") == "naira don rise"

    def test_returns_string(self):
        assert isinstance(clean_text("any text"), str)

    def test_handles_nan(self):
        assert isinstance(clean_text(float('nan')), str)


class TestPreprocess:

    def setup_method(self):
        self.df = pd.DataFrame({
            'tweet': [
                'Naira don rise @CBNgov!!!',
                'things are bad https://t.co/abc',
                'e go better for us',
                '@mention #hashtag'          # will become empty after cleaning
            ],
            'label': ['positive', 'negative', 'positive', 'negative']
        })

    def test_returns_dataframe_and_encoder(self):
        result, encoder = preprocess(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_tweet_column_is_cleaned(self):
        result, _ = preprocess(self.df)
        for tweet in result['tweet']:
            assert '@' not in tweet
            assert 'http' not in tweet

    def test_empty_tweets_dropped(self):
        result, _ = preprocess(self.df)
        assert all(result['tweet'].str.strip() != '')

    def test_label_encoded_column_exists(self):
        result, _ = preprocess(self.df)
        assert 'label_encoded' in result.columns

    def test_label_encoded_is_binary(self):
        result, _ = preprocess(self.df)
        assert set(result['label_encoded'].unique()).issubset({0, 1})

    def test_does_not_modify_original(self):
        original = self.df.copy()
        preprocess(self.df)
        pd.testing.assert_frame_equal(self.df, original)

    def test_index_is_reset(self):
        result, _ = preprocess(self.df)
        assert result.index.tolist() == list(range(len(result)))