# API DOCUMENTATION

Tài liệu API cho Hệ thống Nhận dạng Phương tiện và Biển số xe.

## Base URL

```
http://127.0.0.1:5000
```

## Endpoints

### 1. GET /

Trang chủ - Hiển thị form upload và danh sách file đã xử lý.

**Response**: HTML page

---

### 2. POST /upload

Upload và xử lý file ảnh/video.

**Content-Type**: `multipart/form-data`

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | File ảnh hoặc video |
| cam | String | No | Tên camera (mặc định: "cam1") |
| road | String | No | Tên đường (mặc định: "noname") |
| date | String | No | Ngày (YYYYMMDD, mặc định: ngày hiện tại) |
| time | String | No | Giờ (HHMMSS, mặc định: giờ hiện tại) |

**Supported file formats**:
- Images: `.jpg`, `.jpeg`, `.png`
- Videos: `.mp4`, `.avi`, `.mov`, `.mkv`

**Max file size**: 500MB

**Success Response** (200):
```json
{
  "id": "cam1_noname_20260127_143000"
}
```

**Error Responses**:

400 - Bad Request:
```json
{
  "error": "Chưa chọn file"
}
```

409 - Conflict (duplicate ID):
```json
{
  "error": "File/job với ID này đã tồn tại",
  "id": "cam1_noname_20260127_143000"
}
```

413 - File Too Large:
```json
{
  "error": "File quá lớn. Kích thước tối đa: 500MB"
}
```

---

### 3. GET /status/{id}

Kiểm tra trạng thái xử lý của file.

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | File ID (từ response của /upload) |

**Success Response** (200):

Processing:
```json
{
  "status": "processing",
  "progress": 50,
  "message": "Đang xử lý frame 100/200"
}
```

Done:
```json
{
  "status": "done",
  "progress": 100
}
```

Failed:
```json
{
  "status": "failed",
  "progress": 0,
  "error": "Error message"
}
```

**Error Response** (404):
```json
{
  "error": "Không tìm thấy trạng thái"
}
```

---

### 4. GET /results/{id}

Xem kết quả xử lý chi tiết.

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | String | Yes | File ID |

**Response**: HTML page với:
- Ảnh/video đã được đánh dấu
- Danh sách phương tiện
- Ảnh biển số và kết quả OCR

**Error Response** (404):
```
Không tìm thấy kết quả
```

---

## Examples

### Upload file using cURL

```bash
curl -X POST http://127.0.0.1:5000/upload \
  -F "file=@/path/to/image.jpg" \
  -F "cam=cam1" \
  -F "road=NguyenHuuTho" \
  -F "date=20260127" \
  -F "time=143000"
```

### Check status using cURL

```bash
curl http://127.0.0.1:5000/status/cam1_NguyenHuuTho_20260127_143000
```

### Upload file using Python

```python
import requests

url = "http://127.0.0.1:5000/upload"
files = {"file": open("image.jpg", "rb")}
data = {
    "cam": "cam1",
    "road": "NguyenHuuTho",
    "date": "20260127",
    "time": "143000"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```
