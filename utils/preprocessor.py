
"""
=============================================================================
PREPROCESSOR - Module xử lý text preprocessing
=============================================================================

CHỨC NĂNG:
- Clean text trước khi predict
- Validate input 
- Format text
"""

import re


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
