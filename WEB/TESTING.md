# TESTING GUIDE

Hướng dẫn chạy tests cho hệ thống.

## Cài đặt pytest

```bash
pip install pytest pytest-cov
```

## Chạy tests

### Chạy tất cả tests

```bash
python -m pytest tests/ -v
```

### Chạy test cụ thể

```bash
# Test processor
python -m pytest tests/test_processor.py -v

# Test main
python -m pytest tests/test_main.py -v
```

### Chạy với coverage

```bash
python -m pytest tests/ --cov=main --cov-report=html
```

Kết quả coverage sẽ được lưu trong thư mục `htmlcov/`. Mở file `htmlcov/index.html` để xem chi tiết.

## Cấu trúc tests

```
tests/
├── __init__.py
├── test_processor.py    # Test các hàm xử lý biển số
└── test_main.py         # Test các hàm utility và routes
```

## Lưu ý

- Tests không yêu cầu Redis hoặc models thực tế
- Một số tests sử dụng mock để tránh phụ thuộc vào external services
- Tests có thể cần điều chỉnh nếu có thay đổi trong code

## Manual Testing

### Test upload ảnh

1. Khởi động server: `python main.py`
2. Truy cập: http://127.0.0.1:5000
3. Upload file ảnh test từ `main/data/`
4. Kiểm tra kết quả

### Test upload video

1. Upload file video test từ `main/data/`
2. Đợi xử lý hoàn tất
3. Kiểm tra video output và kết quả tracking

### Test API

Sử dụng cURL hoặc Postman để test các endpoints (xem API_DOCUMENTATION.md)
