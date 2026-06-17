"""
=============================================================================
PREPROCESSOR - Module xử lý text preprocessing
=============================================================================

CHỨC NĂNG:
- Clean text trước khi predict
- Validate input 
- Format text
- Detect language
"""

import re


def detect_vietnamese(text: str) -> bool:
    """
    Detect if text contains Vietnamese characters
    
    Args:
        text: Input text
    
    Returns:
        True if Vietnamese detected, False otherwise
    """
    if not text:
        return False
    
    # Vietnamese diacritics and special characters
    vietnamese_chars = [
        'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
        'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
        'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
        'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
        'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
        'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ',
        'đ', 'Đ'
    ]
    
    # Check if any Vietnamese character exists
    for char in vietnamese_chars:
        if char in text.lower():
            return True
    
    return False


def get_vietnamese_ratio(text: str) -> float:
    """
    Calculate ratio of Vietnamese words in text
    
    Args:
        text: Input text
    
    Returns:
        Ratio of Vietnamese words (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    vietnamese_words = sum(1 for word in words if detect_vietnamese(word))
    
    return vietnamese_words / len(words)


def clean_text(text):
    """
    Clean text using the same method as data_loader.py
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Reuters tags (same as data_loader)
    text = re.sub(r"^.*?\(reuters\)\s*[-–]\s*", "", text, flags=re.IGNORECASE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # ← Thêm 0-9 để giữ số
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class TextPreprocessor:
    """
    Class xử lý preprocessing cho text input
    """
    
    def __init__(self):
        """
        Khởi tạo preprocessor
        """
        pass
    
    def validate_text(self, text: str):
        """
        Validate text input
        """
        pass
    
    def clean_text(self, text: str):
        """
        Clean text sử dụng DataLoader
        """
        pass
    
    def get_text_stats(self, text: str):
        """
        Lấy thống kê về text
        """
        pass
    
    def format_for_display(self, text: str, max_length: int = 100):
        """
        Format text để hiển thị (truncate + ellipsis)
        """
        pass


"""
USAGE:

1. Khởi tạo:
   preprocessor = TextPreprocessor()

2. Validate:
   is_valid, error = preprocessor.validate_text(user_input)
   if not is_valid:
       st.error(error)
       return

3. Clean:
   cleaned = preprocessor.clean_text(user_input)

4. Get stats:
   stats = preprocessor.get_text_stats(user_input)
   st.write(f"Số từ: {stats['word_count']}")

5. Format for display:
   short_text = preprocessor.format_for_display(text, 40)
   st.write(short_text)  # "Đây là một bài báo rất dài..."
"""
