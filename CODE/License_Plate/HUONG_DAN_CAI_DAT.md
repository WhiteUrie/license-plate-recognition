# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY CHƯƠNG TRÌNH

## 1. GIỚI THIỆU

Hệ thống nhận diện biển số xe Việt Nam sử dụng YOLOv8, hỗ trợ nhận diện biển số xe máy và xe ô tô từ ảnh, video hoặc camera real-time.

## 2. YÊU CẦU HỆ THỐNG

- Python 3.8 trở lên
- RAM tối thiểu 4GB
- Webcam (cho chế độ real-time)

## 3. CÀI ĐẶT

### 3.1. Kiểm tra Python

```bash
python --version
```

### 3.2. Tạo môi trường ảo

```bash
# Tạo môi trường
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/macOS)
source venv/bin/activate
```

### 3.3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 4. CẤU TRÚC THƯ MỤC SOURCE CODE

```
license-plate-detection/
│
├── model/                      # Thư mục chứa các file model
│   ├── best1.pt               # Model phát hiện biển số (backup)
│   ├── best2.pt               # Model phát hiện biển số (chính)
│   ├── OCR.pt                 # Model OCR (backup)
│   └── OCR_best.pt            # Model OCR (chính)
│
├── test_data/                  # Thư mục chứa dữ liệu test
│   ├── *.jpg, *.png           # Ảnh test
│   └── video/                 # Video test
│
├── test_results/               # Thư mục lưu kết quả
│   ├── *_result.jpg           # Ảnh kết quả
│   └── *.mp4                  # Video kết quả
│
├── comparison_results/         # Kết quả so sánh
│
├── main.py                     # Ứng dụng GUI chính (real-time)
├── test_image.py               # Script test với ảnh tĩnh
├── test_video.py               # Script test với video
├── yolo_ocr.py                 # Module OCR sử dụng YOLOv8
├── utils.py                    # Các hàm tiện ích xử lý text
├── requirements.txt            # Danh sách thư viện
└── README.md                   # Tài liệu tổng quan
```

### 4.1. Mô tả chức năng các file chính

- **main.py**: Ứng dụng GUI với Tkinter, hỗ trợ nhận diện real-time từ webcam hoặc camera IP
- **test_image.py**: Script để test nhận diện trên ảnh tĩnh, lưu kết quả vào thư mục test_results
- **test_video.py**: Script để test nhận diện trên video, hỗ trợ lưu video kết quả
- **yolo_ocr.py**: Module chứa class YOLOCharacterRecognizer để nhận diện ký tự trên biển số
- **utils.py**: Module chứa các class xử lý và format text biển số (MotorbikeTextProcessor, CarTextProcessor, UnifiedTextProcessor)

## 5. CHUẨN BỊ DỮ LIỆU VÀ MODEL

### 5.1. Vị trí đặt file model

Các file model (.pt) phải được đặt trong thư mục **model/**:

- **best2.pt**: Model phát hiện vị trí biển số xe (bắt buộc)
- **OCR_best.pt**: Model nhận diện ký tự OCR (bắt buộc)

Lưu ý: Đảm bảo 2 file model này tồn tại trước khi chạy chương trình.

### 5.2. Vị trí đặt dữ liệu đầu vào

#### 5.2.1. Ảnh test

Đặt các file ảnh (.jpg, .png) vào thư mục **test_data/**

Ví dụ:

```
test_data/cam1.jpg
test_data/motobike.jpg
test_data/Hue.png
```

#### 5.2.2. Video test

Đặt các file video (.mp4, .avi) vào thư mục **test_data/video/**

Ví dụ:

```
test_data/video/CameraPTZ.mp4
test_data/video/traffic.mp4
```

### 5.3. Quy ước tên file

- Tên file không nên chứa ký tự đặc biệt hoặc khoảng trắng
- Khuyến nghị sử dụng chữ thường và dấu gạch dưới: cam1.jpg, motobike_test.jpg
- Định dạng ảnh hỗ trợ: .jpg, .jpeg, .png
- Định dạng video hỗ trợ: .mp4, .avi, .mov

## 5. CHẠY CHƯƠNG TRÌNH

### 5.1. Nhận diện real-time (GUI)

```bash
python main.py
```

Sử dụng:

- Nhấn "Bắt đầu" để khởi động camera
- Nhấn "Dừng" để dừng camera

### 5.2. Test với ảnh

```bash
python test_image.py
```

Cấu hình đường dẫn ảnh trong file **test_image.py** (dòng 18):

```python
IMAGE_PATH = 'test_data/cam1.jpg'
```

Kết quả lưu tại: **test_results/cam1_result.jpg**

### 5.3. Test với video

```bash
python test_video.py
```

Cấu hình đường dẫn video trong file **test_video.py** (dòng 28):

```python
cap = cv2.VideoCapture('test_data/video/CameraPTZ.mp4')
```

Điều khiển: Nhấn 'q' để thoát, 'p' để tạm dừng

## 7. KẾT QUẢ ĐẦU RA

### 7.1. Kết quả từ ứng dụng GUI (main.py)

- Hiển thị video real-time với bounding box và text biển số
- Màu sắc phân biệt loại xe:
  - **Xanh lá (Green)**: Xe máy
  - **Xanh dương (Blue)**: Xe ô tô
  - **Xám (Gray)**: Không xác định
- Log kết quả nhận diện trong khung "Kết quả nhận diện" với timestamp
- Hiển thị FPS (Frames Per Second) ở góc dưới phải

### 7.2. Kết quả từ test ảnh (test_image.py)

- File ảnh kết quả được lưu trong thư mục **test_results/**
- Tên file: **[tên_ảnh_gốc]\_result.jpg**
- Ảnh có vẽ bounding box và text biển số
- Console hiển thị thông tin: loại xe và biển số

Ví dụ output console:

```
Ảnh: 1920x1080
Đang detect biển số...
Xe may: 75-H1-56374
Car: 51B-12345
Đã lưu kết quả vào: test_results/cam1_result.jpg
```

### 7.3. Kết quả từ test video (test_video.py)

- Hiển thị video real-time với bounding box và text biển số
- Nếu SAVE_VIDEO = True, video kết quả được lưu vào file **video_result.mp4**
- Console hiển thị thông tin FPS và số frame đã xử lý

Ví dụ output console:

```
Đang load models...
Video: 1920x1080 @ 30.0 FPS
Sẽ lưu video kết quả vào: video_result.mp4
Nhấn 'q' để thoát, 'p' để pause/resume
Video kết quả đã lưu tại: video_result.mp4
Đã xử lý 900 frames trong 45.2s (19.9 FPS)
```

### 7.4. Vị trí lưu kết quả

- Ảnh kết quả: **test_results/**
- Video kết quả: **video_result.mp4** (hoặc đường dẫn tùy chỉnh)
- Log console: Hiển thị trực tiếp trên terminal

