"""
Text cleaning utilities - Xử lý làm sạch text trước khi đưa vào model
"""
import re
import string
from typing import Type


class BaseTextCleaner:
    """Common cleaning steps used by all model-specific cleaners."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_urls(text: str) -> str:
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    @staticmethod
    def remove_emails(text: str) -> str:
        return re.sub(r'\S+@\S+', '', text)

    @staticmethod
    def remove_mentions_hashtags(text: str) -> str:
        return re.sub(r'@[\w_]+|#[\w_]+', '', text)

    @staticmethod
    def remove_html(text: str) -> str:
        return re.sub(r'<.*?>', '', text)

    @staticmethod
    def remove_special_chars(text: str, keep: str = '') -> str:
        pattern = rf'[^{re.escape(string.ascii_letters + string.digits + keep)}\s]'
        return re.sub(pattern, ' ', text)

    @classmethod
    def clean_common(cls, text: str) -> str:
        if not text:
            return ""
        text = cls.remove_urls(text)
        text = cls.remove_emails(text)
        text = cls.remove_mentions_hashtags(text)
        text = cls.remove_html(text)
        text = cls.remove_special_chars(text, keep='!?')
        return cls.normalize_whitespace(text)


class MLTextCleaner(BaseTextCleaner):
    """Cleaner for traditional ML models."""

    @classmethod
    def transform(cls, text: str) -> str:
        cleaned = cls.clean_common(text)
        cleaned = cleaned.lower()
        cleaned = cls.remove_special_chars(cleaned, keep='')
        return cls.normalize_whitespace(cleaned)


class LSTMTextCleaner(BaseTextCleaner):
    """Cleaner for LSTM models."""

    @classmethod
    def transform(cls, text: str) -> str:
        cleaned = cls.clean_common(text)
        cleaned = cleaned.lower()
        cleaned = cls.remove_special_chars(cleaned, keep='!?')
        return cls.normalize_whitespace(cleaned)


class BERTTextCleaner(BaseTextCleaner):
    """Cleaner for BERT models."""

    @classmethod
    def transform(cls, text: str) -> str:
        cleaned = cls.clean_common(text)
        return cls.normalize_whitespace(cleaned)


_MODEL_CLEANERS = {
    'logistic': MLTextCleaner,
    'linear_svc': MLTextCleaner,
    'ml': MLTextCleaner,
    'lstm': LSTMTextCleaner,
    'bert': BERTTextCleaner,
}


def get_text_cleaner(model_name: str):
    if not model_name:
        raise ValueError('Model name is required for text cleaning')
    cleaner = _MODEL_CLEANERS.get(model_name.lower())
    if cleaner is None:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(_MODEL_CLEANERS.keys())}")
    return cleaner


def clean_text(text: str, model_name: str = 'bert') -> str:
    """Clean text according to the target model pipeline."""
    cleaner = get_text_cleaner(model_name)
    return cleaner.transform(text)


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from text
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text: str, stopwords: list = None) -> str:
    """
    Remove stopwords from text

    Args:
        text: Input text
        stopwords: List of stopwords to remove
    """
    if stopwords is None:
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']

    words = text.split()
    filtered = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered)
