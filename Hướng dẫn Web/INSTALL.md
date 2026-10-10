# HƯỚNG DẪN CÀI ĐẶT CHI TIẾT

Tài liệu này hướng dẫn từng bước cài đặt hệ thống từ đầu, phù hợp cho người mới bắt đầu.

## Mục lục

1. [Cài đặt Python](#1-cài-đặt-python)
2. [Cài đặt Git](#2-cài-đặt-git)
3. [Clone dự án](#3-clone-dự-án)
4. [Tạo môi trường ảo](#4-tạo-môi-trường-ảo)
5. [Cài đặt thư viện Python](#5-cài-đặt-thư-viện-python)
6. [Cài đặt Redis](#6-cài-đặt-redis)
7. [Cài đặt FFmpeg](#7-cài-đặt-ffmpeg)
8. [Chuẩn bị models](#8-chuẩn-bị-models)
9. [Chạy ứng dụng](#9-chạy-ứng-dụng)
10. [Xử lý lỗi thường gặp](#10-xử-lý-lỗi-thường-gặp)

---

## 1. Cài đặt Python

### Windows

1. Tải Python từ: https://www.python.org/downloads/
2. Chọn phiên bản **Python 3.10** hoặc **3.11** (khuyến nghị)
3. Chạy file cài đặt
4. **QUAN TRỌNG**: Tick vào "Add Python to PATH"
5. Click "Install Now"
6. Kiểm tra cài đặt:

```bash
python --version
```

Kết quả mong đợi: `Python 3.10.x` hoặc `Python 3.11.x`

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

Kiểm tra:
```bash
python3 --version
```

### macOS

```bash
brew install python@3.10
```

Kiểm tra:
```bash
python3 --version
```

---

## 2. Cài đặt Git

### Windows

1. Tải Git từ: https://git-scm.com/download/win
2. Chạy file cài đặt với các tùy chọn mặc định
3. Kiểm tra:

```bash
git --version
```

### Linux

```bash
sudo apt install git
```

### macOS

```bash
brew install git
```

---

## 3. Clone dự án

Mở Terminal/Command Prompt và chạy:

```bash
# Nếu có repository trên GitHub/GitLab
git clone <repository-url>
cd doan

# Hoặc nếu có file zip
# Giải nén file zip và cd vào thư mục
```

---

## 4. Tạo môi trường ảo

Môi trường ảo giúp cách ly các thư viện của dự án, tránh xung đột.

### Windows

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt
venv\Scripts\activate
```

Sau khi kích hoạt, bạn sẽ thấy `(venv)` ở đầu dòng lệnh.

### Linux/macOS

```bash
# Tạo môi trường ảo
python3 -m venv venv

# Kích hoạt
source venv/bin/activate
```

**Lưu ý**: Mỗi lần mở Terminal mới, bạn cần kích hoạt lại môi trường ảo.

---

## 5. Cài đặt thư viện Python

Đảm bảo môi trường ảo đã được kích hoạt (có `(venv)` ở đầu dòng).

```bash
# Upgrade pip
pip install --upgrade pip

# Cài đặt tất cả thư viện
pip install -r requirements.txt
```

Quá trình này có thể mất 5-10 phút tùy vào tốc độ mạng.

### Nếu gặp lỗi với PyTorch

Nếu bạn có GPU NVIDIA và muốn dùng CUDA:

```bash
# Gỡ PyTorch cũ
pip uninstall torch torchvision torchaudio

# Cài PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Nếu chỉ dùng CPU:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

---

## 6. Cài đặt Redis

Redis dùng để lưu trạng thái xử lý (progress, status).

### Windows

**Cách 1: Dùng Redis for Windows (khuyến nghị)**

1. Tải từ: https://github.com/microsoftarchive/redis/releases
2. Tải file `Redis-x64-3.0.504.zip`
3. Giải nén vào thư mục (ví dụ: `C:\Redis`)
4. Mở Command Prompt **với quyền Administrator**
5. Chạy:

```bash
cd C:\Redis
redis-server.exe
```

Giữ cửa sổ này mở khi chạy ứng dụng.

**Cách 2: Dùng WSL (Windows Subsystem for Linux)**

```bash
wsl --install
# Sau khi cài WSL
wsl
sudo apt update
sudo apt install redis-server
redis-server
```

### Linux

```bash
sudo apt update
sudo apt install redis-server

# Khởi động Redis
sudo systemctl start redis

# Kiểm tra
redis-cli ping
# Kết quả: PONG
```

### macOS

```bash
brew install redis

# Khởi động Redis
brew services start redis

# Kiểm tra
redis-cli ping
# Kết quả: PONG
```

---

## 7. Cài đặt FFmpeg

FFmpeg dùng để convert video sang định dạng web-playable.

### Windows

**Cách 1: Dùng Chocolatey (khuyến nghị)**

```bash
# Cài Chocolatey (nếu chưa có)
# Mở PowerShell với quyền Administrator và chạy:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Cài FFmpeg
choco install ffmpeg
```

**Cách 2: Cài thủ công**

1. Tải từ: https://ffmpeg.org/download.html
2. Chọn "Windows builds from gyan.dev"
3. Tải file `ffmpeg-release-essentials.zip`
4. Giải nén vào `C:\ffmpeg`
5. Thêm vào PATH:
   - Mở "Environment Variables"
   - Thêm `C:\ffmpeg\bin` vào PATH
6. Khởi động lại Command Prompt
7. Kiểm tra:

```bash
ffmpeg -version
```

### Linux

```bash
sudo apt update
sudo apt install ffmpeg

# Kiểm tra
ffmpeg -version
```

### macOS

```bash
brew install ffmpeg

# Kiểm tra
ffmpeg -version
```

---

## 8. Chuẩn bị models

Đảm bảo các file model có trong thư mục `main/models/`:

```
main/models/
├── yolov8n.pt      # Model detect phương tiện
├── best.pt         # Model detect biển số
└── OCR_best.pt     # Model OCR ký tự
```

**Nếu thiếu models:**

- Liên hệ người hướng dẫn để lấy file models
- Hoặc tải từ link được cung cấp
- Copy vào thư mục `main/models/`

**Kích thước models:**
- `yolov8n.pt`: ~6MB
- `best.pt`: ~6MB
- `OCR_best.pt`: ~6MB

---

## 9. Chạy ứng dụng

### Bước 1: Khởi động Redis

**Windows:**
```bash
# Mở Command Prompt mới
cd C:\Redis
redis-server.exe
```

**Linux/macOS:**
```bash
# Redis đã chạy tự động nếu dùng systemctl/brew services
# Hoặc chạy thủ công:
redis-server
```

### Bước 2: Khởi động Flask app

Mở Terminal/Command Prompt mới:

```bash
# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Chuyển vào thư mục main
cd main

# Chạy ứng dụng
python main.py
```

Bạn sẽ thấy:
```
🚀 Server running at http://127.0.0.1:5000
 * Serving Flask app 'main'
 * Debug mode: on
```

### Bước 3: Truy cập web

Mở trình duyệt và truy cập: **http://127.0.0.1:5000**

---

## 10. Xử lý lỗi thường gặp

### Lỗi: "redis.exceptions.ConnectionError"

**Nguyên nhân**: Redis chưa chạy

**Giải pháp**:
```bash
# Kiểm tra Redis
redis-cli ping

# Nếu không có kết quả, khởi động Redis
redis-server
```

### Lỗi: "ModuleNotFoundError: No module named 'xxx'"

**Nguyên nhân**: Thiếu thư viện hoặc môi trường ảo chưa kích hoạt

**Giải pháp**:
```bash
# Kích hoạt môi trường ảo
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Cài lại thư viện
pip install -r requirements.txt
```

### Lỗi: "FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'"

**Nguyên nhân**: FFmpeg chưa cài hoặc chưa có trong PATH

**Giải pháp**:
```bash
# Kiểm tra FFmpeg
ffmpeg -version

# Nếu không có, cài lại FFmpeg (xem mục 7)
```

### Lỗi: "RuntimeError: Couldn't load custom C++ ops"

**Nguyên nhân**: Vấn đề với PyTorch/CUDA

**Giải pháp**:
```bash
# Cài lại PyTorch (CPU version)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Lỗi: "Address already in use"

**Nguyên nhân**: Port 5000 đã được sử dụng

**Giải pháp**:
```bash
# Tìm và kill process đang dùng port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/macOS:
lsof -ti:5000 | xargs kill -9

# Hoặc đổi port trong main.py:
# app.run(debug=True, port=5001)
```

### Lỗi: "CUDA out of memory"

**Nguyên nhân**: GPU không đủ RAM

**Giải pháp**:
- Giảm batch size
- Hoặc chuyển sang dùng CPU (tự động fallback)

### Video không phát được trên trình duyệt

**Nguyên nhân**: Video chưa được convert đúng format

**Giải pháp**:
- Kiểm tra FFmpeg đã cài đặt
- Xem log tại `main/logs/app.log`
- Thử với file video khác

### Ứng dụng chạy chậm

**Giải pháp**:
- Nếu có GPU NVIDIA: Cài CUDA và PyTorch với CUDA
- Giảm độ phân giải video/ảnh input
- Tăng RAM cho máy

---

## Kiểm tra cài đặt hoàn chỉnh

Chạy script kiểm tra:

```bash
python -c "
import cv2
import torch
import redis
import flask
from ultralytics import YOLO
print('✅ Tất cả thư viện đã cài đặt thành công!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

Nếu không có lỗi → Cài đặt thành công!

---

## Tóm tắt các lệnh cần chạy

```bash
# 1. Clone dự án
git clone <repo-url>
cd doan

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Khởi động Redis (terminal riêng)
redis-server

# 5. Chạy ứng dụng
cd main
python main.py

# 6. Truy cập http://127.0.0.1:5000
```

---

## Hỗ trợ

Nếu gặp vấn đề không có trong tài liệu:

1. Kiểm tra log tại `main/logs/app.log`
2. Tìm kiếm lỗi trên Google/Stack Overflow
3. Liên hệ người hướng dẫn

---

**Chúc bạn cài đặt thành công! 🎉**
