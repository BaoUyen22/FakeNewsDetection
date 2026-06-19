"""
Text cleaning utilities - Xử lý làm sạch text trước khi đưa vào model
"""
import re
import string


def clean_text(text: str) -> str:
    """
    Làm sạch text cho model prediction
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags (Twitter-style)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove numbers (optional - comment out if needed)
    # text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


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
        # Basic English stopwords
        stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
    
    words = text.split()
    filtered = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered)
