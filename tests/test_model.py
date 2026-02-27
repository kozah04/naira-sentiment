import pytest # type: ignore
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Skipping model tests by default to avoid large HuggingFace download. Set RUN_MODEL_TESTS=1 to enable."
)

from src.model import load_model, predict


@pytest.fixture(scope='module')
def model_and_tokenizer():
    """
    Load model and tokenizer once for all tests in this module.
    scope='module' means it loads once and is reused across all tests.
    """
    model, tokenizer = load_model()
    return model, tokenizer


class TestLoadModel:

    def test_model_loads(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_model_in_eval_mode(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        assert not model.training


class TestPredict:

    def test_returns_list(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        results = predict(["naira don rise"], model, tokenizer)
        assert isinstance(results, list)

    def test_correct_number_of_results(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        tweets = ["naira don rise", "things hard for we", "e go better"]
        results = predict(tweets, model, tokenizer)
        assert len(results) == 3

    def test_result_has_correct_keys(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        results = predict(["naira don rise"], model, tokenizer)
        assert set(results[0].keys()) == {'tweet', 'sentiment', 'confidence'}

    def test_sentiment_is_valid_label(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        results = predict(["naira don rise"], model, tokenizer)
        assert results[0]['sentiment'] in {'positive', 'negative'}

    def test_confidence_is_between_0_and_1(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        results = predict(["naira don rise"], model, tokenizer)
        assert 0.0 <= results[0]['confidence'] <= 1.0

    def test_tweet_preserved_in_output(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        tweet = "naira don rise"
        results = predict([tweet], model, tokenizer)
        assert results[0]['tweet'] == tweet