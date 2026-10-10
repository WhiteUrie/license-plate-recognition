"""
Utilities cho nhận diện biển số xe Việt Nam (xe máy và xe ô tô)
"""

import re
from typing import Tuple, Optional

# Constants chung
OCR_CORRECTIONS = {
    'S': '5', 'I': '1', 'O': '0', 'L': '1', 
    'Z': '2', 'G': '6', 'B': '8', 'T': '7'
}

LETTER_CORRECTIONS = {
    '0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B'
}

PROVINCE_CODES = set(f'{i:02d}' for i in range(11, 100))

MOTORBIKE_LETTERS = {'H', 'T', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 
                     'L', 'N', 'P', 'S', 'U', 'V', 'X', 'Y', 'Z'}

# Xe máy 2 ký tự (biển số đặc biệt) - tất cả tổ hợp từ các ký tự xe máy
MOTORBIKE_EXTENDED_LETTERS = {'A', 'H', 'T', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'L', 'N', 'P', 'S', 'U', 'V', 'X', 'Y', 'Z'}
MOTORBIKE_TWO_LETTERS = {f"{a}{b}" for a in MOTORBIKE_EXTENDED_LETTERS for b in MOTORBIKE_EXTENDED_LETTERS}

# Xe ô tô - tất cả các ký tự
CAR_LETTERS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z'}

# Compiled regex patterns
MOTORBIKE_PATTERN = re.compile(r'^(\d{2})([A-Z])(\d)(\d{4,5})$')
MOTORBIKE_TWO_PATTERN = re.compile(r'^(\d{2})([A-Z]{2})(\d{4,5})$')
CAR_PATTERN = re.compile(r'^(\d{2})([A-Z])(\d{4,5})$')
GENERAL_PATTERN = re.compile(r'^(\d{2})([A-Z]{1,2})(.+)$')


class MotorbikeTextProcessor:
    """Xử lý text cho biển số xe máy Việt Nam (2 hàng)"""
    
    @staticmethod
    def _correct_province(text: str) -> str:
        """Sửa lỗi OCR cho 2 ký tự đầu (mã tỉnh)"""
        if len(text) < 2:
            return text
        result = ""
        for i, char in enumerate(text[:2]):
            if char.isalpha():
                result += OCR_CORRECTIONS.get(char, char)
            else:
                result += char
        return result + text[2:]

    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch và format text xe máy"""
        if not text:
            return ""
        
        text = text.upper().strip()
        text = MotorbikeTextProcessor._correct_province(text)
        text = re.sub(r'[^A-Z0-9\s\-]', '', text)
        
        # Pattern 1: Xe máy 1 ký tự - 75H156374 hoặc 75-H1-56374 -> 75H1-56374
        match = re.match(r'^(\d{2})\s*-?\s*([A-Z])(\d)\s*-?\s*(\d{4,5})$', text)
        if match:
            province, letter, digit, number = match.groups()
            corrected_number = ''.join(OCR_CORRECTIONS.get(c, c) for c in number)
            if letter in MOTORBIKE_LETTERS and len(corrected_number) in [4, 5]:
                return f"{province}{letter}{digit}-{corrected_number}"
        
        # Pattern 2: Xe máy 2 ký tự - 75AB56374 hoặc 75-AB-56374 -> 75AB-56374
        match = re.match(r'^(\d{2})\s*-?\s*([A-Z]{2})\s*-?\s*([A-Z0-9]{4,5})$', text)
        if match:
            province, letters, number = match.groups()
            corrected_number = ''.join(OCR_CORRECTIONS.get(c, c) for c in number)
            if letters in MOTORBIKE_TWO_LETTERS and len(corrected_number) in [4, 5] and corrected_number.isdigit():
                return f"{province}{letters}-{corrected_number}"
        
        return ""
    
    @staticmethod
    def validate(text: str) -> bool:
        """Kiểm tra biển số xe máy hợp lệ"""
        if not text:
            return False
        
        clean = text.replace('-', '')
        
        # Kiểm tra xe máy 1 ký tự
        match = MOTORBIKE_PATTERN.match(clean)
        if match:
            province, letter, _, number = match.groups()
            if (province in PROVINCE_CODES and 
                letter in MOTORBIKE_LETTERS and 
                len(number) in [4, 5]):
                return True
        
        # Kiểm tra xe máy 2 ký tự
        match = MOTORBIKE_TWO_PATTERN.match(clean)
        if match:
            province, letters, number = match.groups()
            if (province in PROVINCE_CODES and 
                letters in MOTORBIKE_TWO_LETTERS and 
                len(number) in [4, 5]):
                return True
        
        return False
    
    @staticmethod
    def calculate_score(text: str) -> float:
        """Tính điểm tin cậy"""
        if not text:
            return 0.0
        
        score = 0.0
        length = len(text.replace('-', ''))
        
        if 8 <= length <= 9:
            score += 40.0
        elif 7 <= length <= 10:
            score += 20.0
        
        if MotorbikeTextProcessor.validate(text):
            score += 50.0
        
        if text[:2] in PROVINCE_CODES:
            score += 10.0
        
        return min(score, 100.0)


class CarTextProcessor:
    """Xử lý text cho biển số xe ô tô Việt Nam (1 hoặc 2 hàng)"""
    
    @staticmethod
    def _correct_province(text: str) -> str:
        """Sửa lỗi OCR cho 2 ký tự đầu (mã tỉnh)"""
        if len(text) < 2:
            return text
        result = ""
        for i, char in enumerate(text[:2]):
            if char.isalpha():
                result += OCR_CORRECTIONS.get(char, char)
            else:
                result += char
        return result + text[2:]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Làm sạch và format text xe ô tô"""
        if not text:
            return ""
        
        text = text.upper().strip()
        text = CarTextProcessor._correct_province(text)
        text = re.sub(r'[^A-Z0-9\s\-\.]', '', text)
        
        # Pattern: 51A12345 hoặc 30A1234
        match = re.match(r'^(\d{2})\s*([A-Z])\s*-?\s*([A-Z0-9]{4,5})$', text)
        if match:
            province, letter, number = match.groups()
            
            if letter in CAR_LETTERS:
                corrected_number = ''.join(OCR_CORRECTIONS.get(c, c) for c in number)
                
                if len(corrected_number) in [4, 5] and corrected_number.isdigit():
                    return f"{province}{letter}-{corrected_number}"
        
        # Pattern đặc biệt: 51BS2345 -> 51B-52345
        match = re.match(r'^(\d{2})([A-Z])([A-Z])([0-9]{4})$', text)
        if match:
            province, letter1, letter2, numbers = match.groups()
            
            if letter1 in CAR_LETTERS:
                corrected_number = OCR_CORRECTIONS.get(letter2, letter2) + numbers
                if corrected_number.isdigit() and len(corrected_number) == 5:
                    return f"{province}{letter1}-{corrected_number}"
        
        return ""
    
    @staticmethod
    def validate(text: str) -> bool:
        """Kiểm tra biển số xe ô tô hợp lệ"""
        if not text:
            return False
        
        clean = text.replace('-', '').replace('.', '').replace(' ', '')
        match = CAR_PATTERN.match(clean)
        
        if match:
            province, letter, number = match.groups()
            return (province in PROVINCE_CODES and 
                    letter in CAR_LETTERS and 
                    len(number) in [4, 5])
        return False
    
    @staticmethod
    def calculate_score(text: str) -> float:
        """Tính điểm tin cậy"""
        if not text:
            return 0.0
        
        score = 0.0
        length = len(text.replace('-', '').replace('.', '').replace(' ', ''))
        
        if 7 <= length <= 8:
            score += 40.0
        elif 6 <= length <= 9:
            score += 20.0
        
        if CarTextProcessor.validate(text):
            score += 50.0
        
        if text[:2] in PROVINCE_CODES:
            score += 10.0
        
        return min(score, 100.0)


class UnifiedTextProcessor:
    """Xử lý text thống nhất cho cả xe máy và xe ô tô"""
    
    @staticmethod
    def _analyze_pattern(clean_text: str) -> str:
        """Phân tích pattern để xác định loại xe"""
        # Xe máy 1 ký tự: 75H156374
        motorbike_match = MOTORBIKE_PATTERN.match(clean_text)
        if motorbike_match:
            _, letter, _, _ = motorbike_match.groups()
            if letter in MOTORBIKE_LETTERS:
                return 'motorbike'
        
        # Xe máy 2 ký tự: 75AB56374
        motorbike_two_match = MOTORBIKE_TWO_PATTERN.match(clean_text)
        if motorbike_two_match:
            _, letters, _ = motorbike_two_match.groups()
            if letters in MOTORBIKE_TWO_LETTERS:
                return 'motorbike'
        
        # Xe máy 2 ký tự với OCR error: 75ABS6374
        motorbike_ocr_match = re.match(r'^(\d{2})([A-Z]{2})([A-Z0-9]{4,5})$', clean_text)
        if motorbike_ocr_match:
            _, letters, numbers = motorbike_ocr_match.groups()
            if letters in MOTORBIKE_TWO_LETTERS:
                corrected_numbers = ''.join(OCR_CORRECTIONS.get(c, c) for c in numbers)
                if corrected_numbers.isdigit() and len(corrected_numbers) in [4, 5]:
                    return 'motorbike'
        
        # Xe ô tô: 51B12345
        car_match = CAR_PATTERN.match(clean_text)
        if car_match:
            _, letter, _ = car_match.groups()
            if letter in CAR_LETTERS:
                return 'car'
        
        return 'unknown'
    
    @staticmethod
    def detect_vehicle_type(text: str, is_two_line: Optional[bool] = None, 
                           line_counts: Optional[Tuple[int, int]] = None) -> str:
        """
        Phát hiện loại xe dựa trên text, layout và số ký tự mỗi hàng
        
        Args:
            text: Text biển số
            is_two_line: Layout từ YOLO (True=2 hàng, False=1 hàng, None=auto)
            line_counts: (line1_count, line2_count) - số ký tự mỗi hàng từ YOLO
        
        Returns: 'motorbike', 'car', hoặc 'unknown'
        
        Logic phân biệt xe máy 2 hàng vs ô tô 2 hàng (cùng 8 ký tự):
        - Xe máy 2 hàng: Hàng 1 có 4 ký tự (61K3), Hàng 2 có 4 ký tự (0615) -> 61-K3-0615
        - Xe ô tô 2 hàng: Hàng 1 có 3 ký tự (61K), Hàng 2 có 5 ký tự (30615) -> 61K-30615
        """
        if not text:
            return 'unknown'
        
        # Chuẩn hóa text
        text = text.upper().strip()
        text = CarTextProcessor._correct_province(text)
        clean_text = text.replace(' ', '').replace('-', '').replace('.', '')
        
        # ===== LOGIC MỚI: Phân biệt dựa trên số ký tự mỗi hàng =====
        if is_two_line and line_counts is not None:
            line1_count, line2_count = line_counts
            total_chars = len(clean_text)
            
            # Biển số 8 ký tự 2 hàng - cần phân biệt xe máy và ô tô
            if total_chars == 8:
                # Xe máy 2 hàng: 4 ký tự hàng 1 + 4 ký tự hàng 2
                # Ví dụ: 61K3 | 0615 -> 61-K3-0615
                if line1_count == 4 and line2_count == 4:
                    return 'motorbike'
                
                # Xe ô tô 2 hàng: 3 ký tự hàng 1 + 5 ký tự hàng 2
                # Ví dụ: 61K | 30615 -> 61K-30615
                if line1_count == 3 and line2_count == 5:
                    return 'car'
            
            # Biển số 9 ký tự 2 hàng
            if total_chars == 9:
                # Xe máy 2 hàng: 4 ký tự hàng 1 + 5 ký tự hàng 2
                # Ví dụ: 61K3 | 06150 -> 61-K3-06150
                if line1_count == 4 and line2_count == 5:
                    return 'motorbike'

        
        # ===== LOGIC CŨ (fallback) =====
        if is_two_line is not None:
            if is_two_line:
                # Layout 2 hàng - ưu tiên xe máy
                result = UnifiedTextProcessor._analyze_pattern(clean_text)
                if result == 'motorbike':
                    return 'motorbike'
                # Xe ô tô 2 hàng (5 số)
                car_match = re.match(r'^(\d{2})([A-Z])(\d{5})$', clean_text)
                if car_match:
                    _, letter_check, _ = car_match.groups()
                    if letter_check in CAR_LETTERS:
                        return 'car'
            else:
                # Layout 1 hàng - ưu tiên xe ô tô
                car_match = CAR_PATTERN.match(clean_text)
                if car_match:
                    _, letter_check, _ = car_match.groups()
                    if letter_check in CAR_LETTERS:
                        return 'car'
                
                # Thử pattern đặc biệt cho OCR error: 51BS2345 -> 51B52345
                special_match = re.match(r'^(\d{2})([A-Z])([A-Z])([0-9]{4})$', clean_text)
                if special_match:
                    province, letter1, letter2, numbers = special_match.groups()
                    if letter1 in CAR_LETTERS and letter2 in OCR_CORRECTIONS:
                        return 'car'
                
                # Fallback to motorbike
                result = UnifiedTextProcessor._analyze_pattern(clean_text)
                if result == 'motorbike':
                    return 'motorbike'
        
        # Không có layout hint - phân tích pattern với ưu tiên xe ô tô
        car_match = CAR_PATTERN.match(clean_text)
        if car_match:
            _, letter_check, _ = car_match.groups()
            if letter_check in CAR_LETTERS:
                return 'car'
        
        return UnifiedTextProcessor._analyze_pattern(clean_text)
    
    @staticmethod
    def process_text(text: str, is_two_line: Optional[bool] = None, 
                    line_counts: Optional[Tuple[int, int]] = None) -> Tuple[str, str, float]:
        """
        Xử lý text và trả về kết quả
        
        Args:
            text: Text biển số
            is_two_line: Layout từ YOLO (optional)
            line_counts: (line1_count, line2_count) từ YOLO (optional)
            
        Returns: (formatted_text, vehicle_type, confidence_score)
        """
        vehicle_type = UnifiedTextProcessor.detect_vehicle_type(text, is_two_line, line_counts)
        
        if vehicle_type == 'motorbike':
            formatted = MotorbikeTextProcessor.clean_text(text)
            score = MotorbikeTextProcessor.calculate_score(formatted)
            return formatted, 'motorbike', score
        
        elif vehicle_type == 'car':
            formatted = CarTextProcessor.clean_text(text)
            score = CarTextProcessor.calculate_score(formatted)
            return formatted, 'car', score
        
        return "", 'unknown', 0.0


# Backward compatibility
TextProcessor = MotorbikeTextProcessor
