# HỆ THỐNG NHẬN DẠNG PHƯƠNG TIỆN VÀ BIỂN SỐ XE

Đồ án tốt nghiệp - Hệ thống nhận dạng tự động phương tiện và biển số xe sử dụng YOLOv8 và Computer Vision.

## Tính năng

- ✅ Nhận dạng phương tiện (xe máy, ô tô, xe tải, xe buýt)
- ✅ Phát hiện và đọc biển số xe Việt Nam
- ✅ Hỗ trợ cả biển số vuông (BSV) và biển số dài (BSD)
- ✅ Xử lý ảnh và video
- ✅ Tracking phương tiện trong video
- ✅ Giao diện web thân thiện
- ✅ Lưu trữ kết quả và lịch sử xử lý

## Công nghệ sử dụng

- **Backend**: Flask (Python)
- **AI/ML**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Database**: Redis
- **Frontend**: HTML, CSS, JavaScript

## Cấu trúc thư mục

```
doan/
├── main/
│   ├── data/              # Dữ liệu mẫu
│   ├── models/            # Models YOLOv8
│   │   ├── yolov8n.pt     # Model detect phương tiện
│   │   ├── best.pt        # Model detect biển số
│   │   └── OCR_best.pt    # Model OCR ký tự
│   ├── static/            # Static files (CSS, images, videos)
│   ├── templates/         # HTML templates
│   ├── uploads/           # Thư mục upload
│   ├── outputs/           # Kết quả xử lý (JSON)
│   ├── logs/              # Log files
│   ├── main.py            # Flask application
│   └── processor.py       # Logic xử lý chính
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── README.md             # File này
└── INSTALL.md            # Hướng dẫn cài đặt chi tiết
```

## Yêu cầu hệ thống

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 - 3.11
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, tăng tốc xử lý)
- **Disk**: 5GB trống

## Cài đặt nhanh

### 1. Clone repository

```bash
git clone <repository-url>
cd doan
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
```

### 3. Kích hoạt môi trường ảo

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 5. Cài đặt Redis

**Windows:**
- Tải Redis từ: https://github.com/microsoftarchive/redis/releases
- Giải nén và chạy `redis-server.exe`

**Linux:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

### 6. Cài đặt FFmpeg (cho xử lý video)

**Windows:**
- Tải từ: https://ffmpeg.org/download.html
- Thêm vào PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 7. Chạy ứng dụng

```bash
cd main
python main.py
```

Truy cập: http://127.0.0.1:5000

## Hướng dẫn sử dụng

### Upload và xử lý file

1. Mở trình duyệt và truy cập http://127.0.0.1:5000
2. Chọn file ảnh hoặc video (định dạng: jpg, png, mp4, avi, mov, mkv)
3. Nhập thông tin:
   - **Camera**: Tên camera (mặc định: cam1)
   - **Đường**: Tên đường (mặc định: noname)
   - **Ngày/Giờ**: Tự động lấy thời gian hiện tại nếu không nhập
4. Nhấn "Tải & Xử lý"
5. Đợi hệ thống xử lý (hiển thị progress bar)
6. Xem kết quả

### Xem kết quả

- Danh sách file đã xử lý hiển thị ở trang chủ
- Click "Xem" để xem chi tiết
- Kết quả bao gồm:
  - Ảnh/video đã được đánh dấu
  - Danh sách phương tiện phát hiện
  - Ảnh biển số và kết quả OCR

### Format tên file tự động

Nếu file có format: `cam1_NguyenHuuTho_20260127_143000.mp4`
- Hệ thống tự động parse: camera, đường, ngày, giờ
- Không cần nhập thủ công

## Format biển số hỗ trợ

### Biển số vuông (BSV)
- `43A 123.45` - 1 chữ cái
- `43-A1 123.45` - 1 chữ + 1 số
- `43-AA 123.45` - 2 chữ cái

### Biển số dài (BSD)
- `43H-191.45` - 2 số + 1 chữ + 5 số

## API Endpoints

### POST /upload
Upload và xử lý file

**Request:**
- `file`: File ảnh/video
- `cam`: Tên camera
- `road`: Tên đường
- `date`: Ngày (YYYYMMDD)
- `time`: Giờ (HHMMSS)

**Response:**
```json
{
  "id": "cam1_noname_20260127_143000"
}
```

### GET /status/{id}
Kiểm tra trạng thái xử lý

**Response:**
```json
{
  "status": "processing",
  "progress": 50,
  "message": "Đang xử lý frame 100/200"
}
```

### GET /results/{id}
Xem kết quả xử lý

## Troubleshooting

### Lỗi: "Cannot connect to Redis"
- Kiểm tra Redis đã chạy: `redis-cli ping` (phải trả về PONG)
- Khởi động Redis: `redis-server`

### Lỗi: "FFmpeg not found"
- Cài đặt FFmpeg và thêm vào PATH
- Kiểm tra: `ffmpeg -version`

### Lỗi: "CUDA not available"
- Không bắt buộc, hệ thống vẫn chạy được trên CPU
- Để dùng GPU: Cài CUDA Toolkit và PyTorch với CUDA

### Video không phát được
- Kiểm tra FFmpeg đã cài đặt
- Thử với file video khác
- Xem log tại `main/logs/app.log`

## Testing

Chạy unit tests:

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
```

Chạy với coverage:

```bash
python -m pytest tests/ --cov=main --cov-report=html
```

## Performance

### Ảnh
- Thời gian xử lý: 1-3 giây/ảnh
- Phụ thuộc: Số lượng phương tiện, độ phân giải

### Video
- Thời gian xử lý: ~1x thời lượng video (realtime)
- Ví dụ: Video 1 phút → xử lý ~1 phút
- Với GPU: Nhanh hơn 2-3 lần

## Giới hạn

- File tối đa: 500MB
- Định dạng hỗ trợ: jpg, jpeg, png, mp4, avi, mov, mkv
- Video tracking: Tối thiểu 10 frames để lưu kết quả

## Tác giả

Đồ án tốt nghiệp - [Tên sinh viên]

## License

[Chọn license phù hợp]

## Liên hệ

- Email: [email]
- GitHub: [github-url]
