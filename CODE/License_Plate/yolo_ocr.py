"""
YOLOv8 OCR - Nhận diện ký tự biển số xe bằng YOLOv8 
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional

class YOLOCharacterRecognizer:
    """Nhận diện ký tự sử dụng YOLOv8"""
    
    def __init__(self, model_path: str = 'characters_best.pt'):
        """
        Khởi tạo model nhận diện ký tự
        
        Args:
            model_path: Đường dẫn đến model YOLOv8 đã train
        """
        self.model = YOLO(model_path)
        
        # Mapping class index sang ký tự dựa trên model thực tế
        self.class_mapping = self._create_class_mapping()
    
    def _create_class_mapping(self) -> dict:
        """Tạo mapping từ class index sang ký tự"""
        chars = '0123456789ABCDEFGHKLMNPSTUVXYZ'
        return {i: char for i, char in enumerate(chars)}
    
    def preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh biển số"""
        # Chuyển sang grayscale nếu cần
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # Resize nếu ảnh quá nhỏ
        height, width = gray.shape
        if width < 200:
            scale = 200 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Tăng contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Chuyển về BGR để YOLO xử lý
        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return bgr
    
    def detect_characters(self, plate_img: np.ndarray, conf_threshold: float = 0.3) -> List[dict]:
        """
        Nhận diện các ký tự trong ảnh biển số
        
        Args:
            plate_img: Ảnh biển số đã cắt
            conf_threshold: Ngưỡng confidence
            
        Returns:
            List các ký tự được detect với thông tin bbox và text
        """
        # Tiền xử lý ảnh
        processed_img = self.preprocess_plate_image(plate_img)
        
        # Detect ký tự
        results = self.model(processed_img, conf=conf_threshold, verbose=False)
        
        characters = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy thông tin
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Chuyển class index sang ký tự
                    char = self.class_mapping.get(cls, '?')
                    
                    # Bỏ qua dấu chấm và các ký tự đặc biệt
                    if char in ['?', '.', '-', ' ']:
                        continue
                    
                    y_center = (y1 + y2) / 2
                    x_center = (x1 + x2) / 2
                    height = y2 - y1
                    
                    characters.append({
                        'char': char,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'x_center': x_center,
                        'y_center': y_center,
                        'height': height
                    })
        
        return characters
    
    def detect_layout(self, characters: List[dict]) -> bool:
        """
        Phát hiện layout chính xác hơn: True=2 hàng, False=1 hàng
        
        Sử dụng clustering thông minh để phân biệt 1 hàng và 2 hàng
        """
        if not characters or len(characters) < 4:
            return False
        
        # Lấy tọa độ y_center của tất cả ký tự
        y_centers = np.array([char['y_center'] for char in characters])
        heights = np.array([char['height'] for char in characters])
        avg_height = np.mean(heights)
        
        # Tính khoảng cách y giữa các ký tự
        y_sorted = np.sort(y_centers)
        y_diffs = np.diff(y_sorted)
        
        # Nếu có khoảng cách lớn hơn 60% chiều cao trung bình -> 2 hàng
        max_gap = np.max(y_diffs) if len(y_diffs) > 0 else 0
        
        # Kiểm tra phân bố y_center
        y_std = np.std(y_centers)
        
        # 2 hàng nếu:
        # 1. Có khoảng cách lớn giữa các nhóm ký tự (gap > 60% avg_height)
        # 2. Độ lệch chuẩn y lớn (> 30% avg_height)
        is_two_line = (max_gap > avg_height * 0.6) or (y_std > avg_height * 0.3)
        
        return is_two_line
    
    def sort_characters(self, characters: List[dict], force_single_line: bool = False) -> Tuple[List[dict], Optional[Tuple[int, int]]]:
        """
        Sắp xếp ký tự theo thứ tự từ trái sang phải
        
        Returns:
            (sorted_characters, line_counts) - line_counts là (line1_count, line2_count) nếu 2 hàng, None nếu 1 hàng
        """
        if not characters:
            return [], None
        
        if force_single_line:
            return sorted(characters, key=lambda c: c['x_center']), None
        
        # Phát hiện layout chính xác
        is_two_line = self.detect_layout(characters)
        
        # Nếu là 1 hàng, sắp xếp đơn giản theo x
        if not is_two_line:
            return sorted(characters, key=lambda c: c['x_center']), None
        
        # Nếu là 2 hàng, dùng thuật toán clustering
        sorted_chars, line1_count, line2_count = self._sort_two_line_plate(characters)
        return sorted_chars, (line1_count, line2_count)
    
    def _sort_two_line_plate(self, characters: List[dict]) -> Tuple[List[dict], int, int]:
        """
        Sắp xếp ký tự cho biển số 2 hàng với xử lý biển số nghiêng
        
        Returns:
            (sorted_characters, line1_count, line2_count)
        """
        if len(characters) < 4:
            return sorted(characters, key=lambda c: c['x_center']), len(characters), 0
        
        # Lấy tọa độ của tất cả ký tự
        y_centers = np.array([char['y_center'] for char in characters])
        x_centers = np.array([char['x_center'] for char in characters])
        
        # === BƯỚC 1: Phát hiện và bù trừ độ nghiêng ===
        # Tính góc nghiêng dựa trên linear regression của tất cả ký tự
        if len(characters) >= 4:
            # Fit đường thẳng y = ax + b
            coeffs = np.polyfit(x_centers, y_centers, 1)
            slope = coeffs[0]  # Độ dốc (tan của góc nghiêng)
            
            # Tính y đã bù trừ nghiêng: y_adjusted = y - slope * x
            # Điều này "xoay" các điểm về ngang
            y_adjusted = y_centers - slope * x_centers
        else:
            y_adjusted = y_centers
            slope = 0
        
        # === BƯỚC 2: Clustering dựa trên y đã bù trừ ===
        from scipy.cluster.vq import kmeans, vq
        
        y_data = y_adjusted.reshape(-1, 1)
        y_min, y_max = np.min(y_adjusted), np.max(y_adjusted)
        initial_centroids = np.array([[y_min], [y_max]])
        
        try:
            centroids, _ = kmeans(y_data.astype(float), initial_centroids.astype(float))
            labels, _ = vq(y_data, centroids)
            
            if centroids[0] < centroids[1]:
                line1_label, line2_label = 0, 1
            else:
                line1_label, line2_label = 1, 0
            
            line1 = [char for i, char in enumerate(characters) if labels[i] == line1_label]
            line2 = [char for i, char in enumerate(characters) if labels[i] == line2_label]
            
        except:
            y_median = np.median(y_adjusted)
            line1 = [char for i, char in enumerate(characters) if y_adjusted[i] < y_median]
            line2 = [char for i, char in enumerate(characters) if y_adjusted[i] >= y_median]
        
        # === BƯỚC 3: Kiểm tra và sửa lỗi phân hàng cho biển số nghiêng lớn ===
        # Nếu độ nghiêng lớn (|slope| > 0.15 ~ 8.5 độ), kiểm tra lại
        if abs(slope) > 0.15 and len(line1) > 0 and len(line2) > 0:
            line1_sorted_temp = sorted(line1, key=lambda c: c['x_center'])
            line2_sorted_temp = sorted(line2, key=lambda c: c['x_center'])
            
            # Kiểm tra ký tự cuối của hàng 2 có x_center lớn hơn ký tự cuối hàng 1 không
            # Nếu có, và y_center của nó gần với hàng 1 hơn -> có thể bị phân sai
            if line2_sorted_temp:
                last_char_line2 = line2_sorted_temp[-1]
                
                # Tính khoảng cách y trung bình của mỗi hàng (đã bù trừ)
                line1_y_avg = np.mean([y_adjusted[characters.index(c)] for c in line1])
                line2_y_avg = np.mean([y_adjusted[characters.index(c)] for c in line2])
                gap = abs(line2_y_avg - line1_y_avg)
                
                # Kiểm tra các ký tự ở rìa phải của hàng 2
                # Nếu chúng có x_center lớn và y gần với hàng 1 -> giữ nguyên ở hàng 2
                chars_to_check = [c for c in line1 if c['x_center'] > np.percentile(x_centers, 60)]
                
                for char in chars_to_check:
                    char_idx = characters.index(char)
                    char_y_adj = y_adjusted[char_idx]
                    
                    # Nếu ký tự này gần hàng 2 hơn (theo y đã bù trừ)
                    dist_to_line1 = abs(char_y_adj - line1_y_avg)
                    dist_to_line2 = abs(char_y_adj - line2_y_avg)
                    
                    # Nếu ký tự ở rìa phải và gần hàng 2 hơn -> chuyển về hàng 2
                    if dist_to_line2 < dist_to_line1 * 0.8:
                        line1.remove(char)
                        line2.append(char)
        
        # === BƯỚC 4: Sắp xếp mỗi hàng theo x ===
        line1_sorted = sorted(line1, key=lambda c: c['x_center'])
        line2_sorted = sorted(line2, key=lambda c: c['x_center'])
        
        return line1_sorted + line2_sorted, len(line1_sorted), len(line2_sorted)
    
    def recognize_plate_text(self, plate_img: np.ndarray, conf_threshold: float = 0.3, 
                            force_single_line: bool = False) -> Tuple[str, float, Optional[Tuple[int, int]]]:
        """
        Nhận diện text từ ảnh biển số
        
        Args:
            plate_img: Ảnh biển số
            conf_threshold: Ngưỡng confidence
            force_single_line: Bắt buộc sắp xếp 1 hàng (cho xe ô tô)
            
        Returns:
            (text, avg_confidence, line_counts) - line_counts là (line1_count, line2_count) nếu 2 hàng
        """
        # Detect ký tự
        characters = self.detect_characters(plate_img, conf_threshold)
        
        if not characters:
            return "", 0.0, None
        
        # Sắp xếp ký tự
        sorted_chars, line_counts = self.sort_characters(characters, force_single_line)
        
        # Ghép thành text
        text = ''.join([char['char'] for char in sorted_chars])
        
        # Tính confidence trung bình
        avg_conf = np.mean([char['confidence'] for char in sorted_chars])
        
        return text, float(avg_conf), line_counts
    
    def recognize_with_multiple_attempts(self, plate_img: np.ndarray, 
                                        force_single_line: bool = False) -> Tuple[str, float, str, bool, Optional[Tuple[int, int]]]:
        """
        Nhận diện với nhiều lần thử và chọn kết quả tốt nhất
        
        Returns: (text, confidence, vehicle_type, is_two_line, line_counts)
            - line_counts: (line1_count, line2_count) để phân biệt xe máy/ô tô 2 hàng
        """
        attempts = [
            ('original', plate_img),
            ('enhanced', cv2.convertScaleAbs(plate_img, alpha=1.5, beta=10)),
            ('blurred', cv2.GaussianBlur(plate_img, (3, 3), 0))
        ]
        
        results = []
        
        for name, img in attempts:
            characters = self.detect_characters(img, conf_threshold=0.25)
            if characters:
                is_two_line = self.detect_layout(characters)
                sorted_chars, line_counts = self.sort_characters(characters, force_single_line)
                text = ''.join([char['char'] for char in sorted_chars])
                conf = np.mean([char['confidence'] for char in sorted_chars])
                results.append((text, conf, is_two_line, line_counts))
        
        if not results:
            return "", 0.0, 'unknown', False, None
        
        # Chọn kết quả có confidence cao nhất
        best_text, best_conf, best_layout, best_line_counts = max(results, key=lambda x: x[1])
        
        # Phát hiện loại xe - truyền thêm line_counts để phân biệt xe máy/ô tô 2 hàng
        from utils import UnifiedTextProcessor
        vehicle_type = UnifiedTextProcessor.detect_vehicle_type(
            best_text, 
            is_two_line=best_layout, 
            line_counts=best_line_counts
        )
        
        return best_text, best_conf, vehicle_type, best_layout, best_line_counts
    
    def visualize_detection(self, plate_img: np.ndarray, characters: List[dict]) -> np.ndarray:
        """Vẽ kết quả detection lên ảnh"""
        result_img = plate_img.copy()
        
        for char_info in characters:
            x1, y1, x2, y2 = char_info['bbox']
            char = char_info['char']
            conf = char_info['confidence']
            
            # Vẽ bbox
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Vẽ text
            label = f"{char} {conf:.2f}"
            cv2.putText(result_img, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result_img