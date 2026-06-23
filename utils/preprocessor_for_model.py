import re
import unicodedata
import pandas as pd
from typing import Tuple


# ============================================================================
# BASE CLEANING - Áp dụng cho TẤT CẢ input trước khi xử lý theo model
# ============================================================================

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
MULTISPACE_PATTERN = re.compile(r"\s+")


def base_clean_text(text: str) -> str:
    """
    Base cleaning - áp dụng cho TẤT CẢ input trước khi preprocessing theo model
    
    Steps:
        1. Normalize Unicode (NFKC)
        2. Remove URLs (http://, www.)
        3. Remove HTML tags
        4. Remove mentions (@username)
        5. Remove hashtags (#tag)
        6. Reduce repeated characters (aaaa → aa)
        7. Normalize whitespace
    
    Args:
        text: Raw input text
    
    Returns:
        Base-cleaned text
    
    Example:
        >>> base_clean_text("OMG!!! https://spam.com <b>SHOCKING</b> @user #fake")
        "OMG!!! SHOCKING"
    """
    if pd.isna(text) or not text:
        return ""
    
    # 1. Normalize Unicode
    value = unicodedata.normalize("NFKC", str(text))
    
    # 2. Remove URLs
    value = URL_PATTERN.sub(" ", value)
    
    # 3. Remove HTML tags
    value = HTML_PATTERN.sub(" ", value)
    
    # 4. Remove mentions
    value = re.sub(r"@[\w_]+", " ", value)
    
    # 5. Remove hashtags
    value = re.sub(r"#[\w_]+", " ", value)
    
    # 6. Reduce repeated characters (e.g., "aaaa" → "aa")
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    
    # 7. Normalize whitespace
    value = MULTISPACE_PATTERN.sub(" ", value).strip()
    
    return value


class BaseModelPreprocessor:
    """Common helpers for model-specific preprocessing."""

    STOPWORDS = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am',
        'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been',
        'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could',
        'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few',
        'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he',
        'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
        'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just',
        'me', 'more', 'most', 'my', 'myself', 'now', 'of', 'off', 'on',
        'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves',
        'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some',
        'such', 'than', 'that', 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
        'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself',
        'yourselves', 'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven',
        'hadn', 'doesn', 'don', 'didn', 'won', 'wouldn', 'shan', 'shouldn',
        'mightn', 'mustn', 'couldn', 'ain', "ain't"
    }

    NEGATIONS = {
        'not', 'no', 'never', 'neither', 'nor', 'none',
        'nobody', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
    }

    INTENSIFIERS = {
        'very', 'really', 'extremely', 'absolutely', 'totally',
        'completely', 'highly', 'super', 'so', 'too', 'quite',
        'rather', 'somewhat', 'especially', 'utterly', 'incredibly',
    }

    @classmethod
    def filtered_stopwords(cls) -> set[str]:
        return cls.STOPWORDS - cls.NEGATIONS - cls.INTENSIFIERS

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def lemmatize_token(token: str) -> str:
        token = token.strip()
        if len(token) <= 3:
            return token
        if token.endswith('ies') and len(token) > 4:
            return token[:-3] + 'y'
        if token.endswith('ves') and len(token) > 4:
            return token[:-3] + 'f'
        if token.endswith('ing') and len(token) > 4:
            stem = token[:-3]
            if stem.endswith('ie'):
                return stem[:-2] + 'y'
            if stem.endswith('e') and len(stem) > 1:
                return stem
            return stem
        if token.endswith('ed') and len(token) > 3:
            stem = token[:-2]
            if stem.endswith('i'):
                return stem[:-1] + 'y'
            return stem
        if token.endswith('s') and len(token) > 2 and not token.endswith(('ss', 'us', 'is', 'ous')):
            return token[:-1]
        return token


class MLTextPreprocessor(BaseModelPreprocessor):
    """Tiền xử lý dành cho Logistic Regression, Linear SVC."""

    @staticmethod
    def transform(text: str) -> Tuple[str, int, int]:
        if not isinstance(text, str) or not text:
            return "", 0, 0
        
        # Step 1: Base cleaning (URLs, HTML, mentions, hashtags, etc.)
        text = base_clean_text(text)
        if not text:
            return "", 0, 0

        # Step 2: Model-specific preprocessing
        lower_text = text.lower()
        count_excl = lower_text.count('!')
        count_quest = lower_text.count('?')

        cleaned = re.sub(r'[^\w\s!?]', ' ', lower_text, flags=re.UNICODE)
        cleaned = re.sub(r'[!?]', ' ', cleaned)
        cleaned = cleaned.replace('_', ' ')
        cleaned = BaseModelPreprocessor.normalize_whitespace(cleaned)

        tokens = [token for token in cleaned.split() if token not in BaseModelPreprocessor.filtered_stopwords()]
        tokens = [BaseModelPreprocessor.lemmatize_token(token) for token in tokens]

        return ' '.join(tokens), count_excl, count_quest


class LSTMTextPreprocessor(BaseModelPreprocessor):
    """Tiền xử lý dành cho LSTM."""

    @staticmethod
    def transform(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        
        # Step 1: Base cleaning (URLs, HTML, mentions, hashtags, etc.)
        text = base_clean_text(text)
        if not text:
            return ""

        # Step 2: Model-specific preprocessing
        lower_text = text.lower()
        cleaned = re.sub(r'[^\w\s!?]', ' ', lower_text, flags=re.UNICODE)
        cleaned = cleaned.replace('_', ' ')
        return BaseModelPreprocessor.normalize_whitespace(cleaned)


class BERTTextPreprocessor(BaseModelPreprocessor):
    """Tiền xử lý dành cho BERT."""

    @staticmethod
    def transform(text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        
        # Step 1: Base cleaning (URLs, HTML, mentions, hashtags, etc.)
        text = base_clean_text(text)
        if not text:
            return ""

        # Step 2: Model-specific preprocessing (minimal for BERT)
        return BaseModelPreprocessor.normalize_whitespace(text)
