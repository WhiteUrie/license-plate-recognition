"""
Test nhận diện biển số xe từ ảnh tĩnh
"""

import cv2
import os
from ultralytics import YOLO
from yolo_ocr import YOLOCharacterRecognizer
from utils import UnifiedTextProcessor


def get_output_path(image_path: str, output_dir: str = 'test_results') -> str:
    """Tạo đường dẫn output không bị ghi đè"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    if not os.path.exists(output_path):
        return output_path
    
    version = 1
    while True:
        output_path = os.path.join(output_dir, f"{base_name}_result_v{version}.jpg")
        if not os.path.exists(output_path):
            return output_path
        version += 1


def main():
    # === CẤU HÌNH ===
    IMAGE_PATH = 'test_data/grap.jpg'
    OUTPUT_PATH = get_output_path(IMAGE_PATH)
    
    # Load models
    print("Đang load models...")
    plate_model = YOLO('model/best2.pt')
    ocr_model = YOLOCharacterRecognizer('model/OCR_best.pt')
    
    # Đọc ảnh
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Không thể đọc ảnh: {IMAGE_PATH}")
        return
    
    print(f"Ảnh: {image.shape[1]}x{image.shape[0]}")
    print("Đang detect biển số...")
    
    # Colors và labels
    COLORS = {
        "motorbike": (0, 255, 0),
        "car": (255, 0, 0),
        "unknown": (128, 128, 128)
    }
    LABELS = {"motorbike": "Xe may", "car": "Car", "unknown": "Unknown"}
    
    # Detect biển số
    results = plate_model(image, verbose=False, imgsz=640)
    
    detections = []
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                if box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    plate_img = image[y1:y2, x1:x2]
                    
                    if plate_img.size > 0:
                        # OCR
                        text, conf, line_counts = ocr_model.recognize_plate_text(
                            plate_img, conf_threshold=0.3
                        )
                        
                        if text:
                            is_two_line = line_counts is not None
                            formatted_text, detected_type, score = \
                                UnifiedTextProcessor.process_text(text, is_two_line, line_counts)
                            
                            if formatted_text:
                                color = COLORS.get(detected_type, COLORS["unknown"])
                                label_prefix = LABELS.get(detected_type, "")
                                label = f"{label_prefix}: {formatted_text}" if label_prefix else formatted_text
                                
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'label': label,
                                    'color': color,
                                    'text': formatted_text,
                                    'type': detected_type,
                                    'conf': conf,
                                    'score': score
                                })
                                
                                print(f"{LABELS.get(detected_type, 'Unknown')}: {formatted_text}")
    
    if not detections:
        print("Không phát hiện được biển số nào!")
        return
    
    # Vẽ kết quả
    result_image = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(result_image, (x1, y1), (x2, y2), det['color'], 2)
        cv2.putText(result_image, det['label'], (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, det['color'], 2)
    
    # Lưu kết quả
    cv2.imwrite(OUTPUT_PATH, result_image)
    print(f"Đã lưu kết quả vào: {OUTPUT_PATH}")
    
    # Hiển thị (optional)
    cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('License Plate Detection', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
