# src.utils package
import threading

# Thread local storage for session ID sharing across components
thread_local = threading.local()

# Export CustomCallBackHandler 
from .callback_handler import CustomCallBackHandler

# Export patient utilities
from .patient_utils import store_patient_details 

# Simple language detection function
def get_language_code(text: str) -> str:
    """
    Detect language of input text using basic heuristics.
    Returns 'ar' for Arabic, 'en' for English (default)
    """
    # Simple Arabic character detection
    arabic_chars = [
        'ا', 'أ', 'إ', 'آ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش',
        'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي', 'ى', 'ة'
    ]
    
    # Count Arabic characters
    arabic_count = sum(1 for c in text if c in arabic_chars)
    
    # If more than 10% of characters are Arabic, assume Arabic
    if arabic_count > len(text) * 0.1:
        return 'ar'
    else:
        return 'en' 