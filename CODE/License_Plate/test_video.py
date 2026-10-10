"""
Test nhận diện biển số xe real-time từ video (Optimized)
"""

import cv2
import time
from ultralytics import YOLO
from yolo_ocr import YOLOCharacterRecognizer
from utils import UnifiedTextProcessor


def main():
    # === CẤU HÌNH ===
    SAVE_VIDEO = True  # True: lưu video output, False: chỉ hiển thị
    OUTPUT_PATH = 'video_result.mp4'
    
    # Load models
    print("Đang load models...")
    plate_model = YOLO('model/best2.pt')
    ocr_model = YOLOCharacterRecognizer('model/OCR_best.pt')
    
    # Mở video
    cap = cv2.VideoCapture('test_data/video/CameraPTZ.mp4')
    if not cap.isOpened():
        print("Không thể mở video!")
        return
    
    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Colors và labels
    COLORS = {
        "motorbike": (0, 255, 0),
        "car": (255, 0, 0),
        "unknown": (128, 128, 128)
    }
    LABELS = {"motorbike": "Xe may", "car": "Car", "unknown": ""}
    
    frame_count = 0
    start_time = time.time()
    
    # === OPTIMIZATION: Cache kết quả OCR ===
    cached_results = {}  # {plate_region_hash: (label, color, bbox)}
    SKIP_FRAMES = 2  # Chỉ detect mỗi N frames
    last_detections = []  # Lưu detection gần nhất để vẽ lên frame bị skip
    
    # Tạo cửa sổ có thể resize
    cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('License Plate Detection', 1280, 720)
    
    # === Tạo VideoWriter để lưu kết quả ===
    out = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"Sẽ lưu video kết quả vào: {OUTPUT_PATH}")
    
    print("Nhấn 'q' để thoát, 'p' để pause/resume")
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # === OPTIMIZATION: Skip frames ===
            if frame_count % SKIP_FRAMES == 0:
                last_detections = []
                
                # Detect biển số với imgsz nhỏ hơn để nhanh hơn
                results = plate_model(frame, verbose=False, imgsz=640)
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            if box.conf[0] > 0.5:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                plate_img = frame[y1:y2, x1:x2]
                                
                                if plate_img.size > 0:
                                    # === OPTIMIZATION: OCR đơn giản, không multiple attempts ===
                                    text, conf, line_counts = ocr_model.recognize_plate_text(
                                        plate_img, conf_threshold=0.3
                                    )
                                    
                                    if text:
                                        is_two_line = line_counts is not None
                                        formatted_text, detected_type, score = \
                                            UnifiedTextProcessor.process_text(text, is_two_line, line_counts)
                                        
                                        if formatted_text and (conf * 100 + score) / 2 >= 50:
                                            color = COLORS.get(detected_type, COLORS["unknown"])
                                            label_prefix = LABELS.get(detected_type, "")
                                            label = f"{label_prefix}: {formatted_text}" if label_prefix else formatted_text
                                            last_detections.append((x1, y1, x2, y2, label, color))
            
            # Vẽ detections
            for x1, y1, x2, y2, label, color in last_detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Ghi frame vào video output
            if SAVE_VIDEO and out is not None:
                out.write(frame)
        
        # Resize frame để hiển thị vừa màn hình
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('License Plate Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
    
    cap.release()
    if out is not None:
        out.release()
        print(f"Video kết quả đã lưu tại: {OUTPUT_PATH}")
    cv2.destroyAllWindows()
    print(f"Đã xử lý {frame_count} frames trong {elapsed:.1f}s ({current_fps:.1f} FPS)")


if __name__ == "__main__":
    main()
