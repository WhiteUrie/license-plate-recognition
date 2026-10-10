import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
from utils import UnifiedTextProcessor
from yolo_ocr import YOLOCharacterRecognizer

class LicensePlateDetector:
    def __init__(self):
        # Khởi tạo YOLOv8 model cho detect biển số
        self.model = YOLO('model/best2.pt')  # Model detect biển số
        
        # Khởi tạo YOLOv8 OCR cho nhận diện ký tự
        self.ocr_model = YOLOCharacterRecognizer('model/OCR_best.pt')
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # GUI
        self.root = None
        self.video_label = None
        self.result_text = None
        self.status_label = None
        self.fps_label = None
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()

        
    def detect_license_plate(self, frame):
        """Detect biển số trong frame"""
        results = self.model(frame, verbose=False)
        license_plates = []
        
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if box.conf[0] > 0.5: # Giảm từ 0.5 xuống 0.2
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        license_plates.append((int(x1), int(y1), int(x2), int(y2)))
        
        return license_plates
        
    def extract_text_from_plate(self, plate_img):
        """Trích xuất text từ ảnh biển số sử dụng YOLOv8 OCR"""
        try:
            raw_text, confidence, vehicle_type, is_two_line, line_counts = self.ocr_model.recognize_with_multiple_attempts(plate_img)
            
            if not raw_text:
                return "", "unknown"
            
            # Truyền line_counts để phân biệt xe máy/ô tô 2 hàng chính xác hơn
            formatted_text, detected_type, score = UnifiedTextProcessor.process_text(raw_text, is_two_line=is_two_line, line_counts=line_counts)
            
            if not formatted_text:
                return "", "unknown"
            
            final_score = (confidence * 100 + score) / 2
            
            if final_score >= 50:
                return formatted_text, detected_type
            
            return "", "unknown"
            
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "", "unknown"
    
    def process_frame(self, frame):
        """Xử lý frame: detect biển số và OCR"""
        plates = self.detect_license_plate(frame)
        detected_results = []
        
        COLORS = {
            "motorbike": (0, 255, 0),      # Xanh lá
            "car": (255, 0, 0),             # Xanh dương
            "unknown": (128, 128, 128)      # Xám
        }
        
        LABELS = {
            "motorbike": "Xe may",
            "car": "Xe o to",
            "unknown": ""
        }
        
        for (x1, y1, x2, y2) in plates:
            plate_img = frame[y1:y2, x1:x2]
            
            if plate_img.size > 0:
                text, vehicle_type = self.extract_text_from_plate(plate_img)
                if text:
                    detected_results.append((text, vehicle_type))
                    
                    color = COLORS.get(vehicle_type, COLORS["unknown"])
                    label_prefix = LABELS.get(vehicle_type, "")
                    label = f"{label_prefix}: {text}" if label_prefix else text
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame, detected_results
    
    def start_camera(self):
        """Khởi động camera"""
        # Dùng webcam (0 = webcam mặc định)
        # Nếu muốn dùng camera IP, thay 0 bằng "http://IP:PORT/video"
        self.cap = cv2.VideoCapture(0)
        # self.cap = cv2.VideoCapture("http://100.125.56.17:8080/video")
        # self.cap = cv2.VideoCapture("video10s.mp4")

        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera!")
            return False
        
        # Cài đặt buffer size để giảm độ trễ
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Giảm độ phân giải để xử lý nhanh hơn (tùy chọn)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Thử đọc frame đầu tiên để kiểm tra
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc dữ liệu từ camera!")
            self.cap.release()
            return False
        
        print(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        self.is_running = True
        return True
    
    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def test_camera_connection(self):
        """Test kết nối camera"""
        self.status_label.configure(text="Đang test camera...", foreground="orange")
        self.root.update()
        
        test_cap = None
        try:
            # Kiểm tra cùng webcam mặc định được sử dụng khi nhận diện.
            test_cap = cv2.VideoCapture(0)
            if test_cap.isOpened():
                ret, frame = test_cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    self.status_label.configure(text=f"Camera OK - {w}x{h}", foreground="green")
                    messagebox.showinfo("Test Camera", f"Kết nối thành công!\nĐộ phân giải: {w}x{h}")
                else:
                    self.status_label.configure(text="Camera không có dữ liệu", foreground="red")
                    messagebox.showerror("Test Camera", "Không thể đọc dữ liệu từ camera")
            else:
                self.status_label.configure(text="Không thể kết nối camera", foreground="red")
                messagebox.showerror("Test Camera", "Không thể kết nối đến webcam mặc định")
        except Exception as e:
            self.status_label.configure(text="Lỗi test camera", foreground="red")
            messagebox.showerror("Test Camera", f"Lỗi: {str(e)}")
        finally:
            if test_cap is not None:
                test_cap.release()
    
    def resize_frame_for_display(self, frame, max_width=640, max_height=480):
        """Resize frame để hiển thị trong GUI giữ nguyên tỷ lệ"""
        height, width = frame.shape[:2]
        
        # Tính tỷ lệ scale
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        # Tính kích thước mới
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized_frame
    
    def update_frame(self):
        """Cập nhật frame trong GUI"""
        if not (self.is_running and self.cap):
            return
        
        # Bỏ qua frames cũ để giảm delay
        for _ in range(5):
            if not self.cap.grab():
                break
        
        ret, frame = self.cap.retrieve()
        
        if ret:
            self.frame_count += 1
            
            # Tính FPS
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 0:
                fps = self.frame_count / elapsed_time
                self.fps_label.configure(text=f"FPS: {fps:.1f}")
            
            # Xử lý và hiển thị frame
            processed_frame, texts = self.process_frame(frame)
            display_frame = self.resize_frame_for_display(processed_frame)
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)
            
            self.video_label.configure(image=frame_tk)
            self.video_label.image = frame_tk
            self.status_label.configure(text="Đang chạy...", foreground="green")
            
            # Cập nhật kết quả
            if texts:
                current_time = time.strftime("%H:%M:%S")
                vehicle_labels = {"motorbike": "Xe máy", "car": "Xe ô tô", "unknown": "Không xác định"}
                
                for text, vehicle_type in texts:
                    label = vehicle_labels.get(vehicle_type, "Không xác định")
                    self.result_text.insert(tk.END, f"[{current_time}] {label}: {text}\n")
                self.result_text.see(tk.END)
        else:
            self.status_label.configure(text="Lỗi đọc camera", foreground="red")
        
        if self.is_running:
            self.root.after(30, self.update_frame)
    
    def create_gui(self):
        """Tạo giao diện GUI"""
        self.root = tk.Tk()
        self.root.title("Hệ thống nhận diện biển số xe Việt Nam (Xe máy & Xe ô tô) - YOLOv8")
        self.root.geometry("1000x700")  # Tăng kích thước cửa sổ
        
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Camera", padding="5")
        video_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Tạo frame container cho video để center nó
        video_container = ttk.Frame(video_frame)
        video_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_container, text="Camera chưa khởi động", 
                                   anchor="center", background="black", foreground="white")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure video container
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        video_container.columnconfigure(0, weight=1)
        video_container.rowconfigure(0, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        start_btn = ttk.Button(control_frame, text="Bắt đầu", command=self.start_detection)
        start_btn.grid(row=0, column=0, padx=(0, 5))
        
        stop_btn = ttk.Button(control_frame, text="Dừng", command=self.stop_detection)
        stop_btn.grid(row=0, column=1, padx=(5, 0))
        
        test_btn = ttk.Button(control_frame, text="Test Camera", command=self.test_camera_connection)
        test_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Kết quả nhận diện", padding="5")
        result_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Text widget với scrollbar
        text_frame = ttk.Frame(result_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.result_text = tk.Text(text_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="Sẵn sàng", foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.grid(row=0, column=1, sticky=tk.E)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)
    
    def start_detection(self):
        """Bắt đầu nhận diện"""
        if self.start_camera():
            # Reset FPS counter
            self.frame_count = 0
            self.start_time = time.time()
            self.update_frame()
        else:
            self.status_label.configure(text="Không thể kết nối camera", foreground="red")
    
    def stop_detection(self):
        """Dừng nhận diện"""
        self.stop_camera()
        self.status_label.configure(text="Đã dừng", foreground="blue")
        self.fps_label.configure(text="FPS: 0")
        
        # Hiển thị thông báo camera đã dừng
        self.video_label.configure(image="", text="Camera đã dừng")
        self.video_label.image = None
    
    def run(self):
        """Chạy ứng dụng"""
        self.create_gui()
        
        # Xử lý sự kiện đóng cửa sổ
        def on_closing():
            self.stop_camera()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    detector = LicensePlateDetector()
    detector.run()