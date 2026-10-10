import os
import cv2
import json
import re
import subprocess
import numpy as np
import logging
from datetime import datetime, timedelta
from ultralytics import YOLO
from collections import Counter

# Logging
logger = logging.getLogger(__name__)

# Helper functions
def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def filter_overlapping_boxes(boxes, iou_threshold=0.4):
    """
    Loại bỏ boxes trùng lặp dựa trên IoU
    Giữ lại box có diện tích lớn nhất
    
    Args:
        boxes: numpy array of boxes [[x1,y1,x2,y2], ...]
        iou_threshold: ngưỡng IoU để coi là trùng lặp (0.3-0.5)
    
    Returns:
        list of filtered boxes
    """
    if len(boxes) == 0:
        return boxes
    
    # Convert to list nếu là numpy array
    boxes = [box for box in boxes]
    
    # Sort theo diện tích (lớn nhất trước)
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    
    keep = []
    while len(boxes) > 0:
        current = boxes[0]
        keep.append(current)
        boxes = boxes[1:]
        
        # Loại bỏ boxes overlap với current
        boxes = [b for b in boxes if calculate_iou(current, b) < iou_threshold]
    
    return keep

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
STATIC_IMG = os.path.join(BASE_DIR, "static", "images")
STATIC_VIDEO = os.path.join(BASE_DIR, "static", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def process_file(path, fid, redis):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        process_image(path, fid, redis)
    elif ext in VIDEO_EXTS:
        process_video(path, fid, redis)
    else:
        redis.set(fid, json.dumps({
            "status": "error",
            "progress": 0,
            "message": "Định dạng file không hỗ trợ"
        }))


for p in [STATIC_IMG, STATIC_VIDEO, OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

# Load models (cached in memory)
vehicle_model = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt"))
plate_crop_model = YOLO(os.path.join(MODEL_DIR, "best.pt"))
plate_ocr_model = YOLO(os.path.join(MODEL_DIR, "OCR_best.pt"))

def merge_tracks_by_plate(vehicles):
    """
    Gộp nhiều track thành 1 vehicle theo plate_final
    """
    merged = {}
    next_idx = 1

    for v in vehicles:
        plate = v.get("plate_final")

        # Không gộp nếu không đọc được biển
        if not plate or plate in ["UNKNOWN", "Không đọc được"]:
            key = f"__NO_PLATE__{id(v)}"
        else:
            key = plate

        if key not in merged:
            merged[key] = {
                "vehicle_index": next_idx,
                "plate_final": plate,
                "vehicle_image": v["vehicle_image"],
                "plate_image": v["plate_image"],
                "frames_seen": v["frames_seen"],
                "track_ids": [v["track_id"]]
            }
            next_idx += 1
        else:
            m = merged[key]
            m["track_ids"].append(v["track_id"])

            # Ưu tiên track thấy lâu hơn
            if v["frames_seen"] > m["frames_seen"]:
                m["vehicle_image"] = v["vehicle_image"]
                m["plate_image"] = v["plate_image"]
                m["frames_seen"] = v["frames_seen"]

    return list(merged.values())

def ocr_plate_layout(plate_img, plate_type="bsv"):
    """
    OCR và xác định layout biển số
    
    Args:
        plate_img: ảnh biển số
        plate_type: "bsv" (vuông - 2 hàng) hoặc "bsd" (dài - 1 hàng)
    
    Returns:
        tuple: (layout, chars_data) hoặc (None, None)
    """
    ocr = plate_ocr_model(plate_img, conf=0.3)
    if not ocr or ocr[0].boxes is None:
        return None, None

    chars = []
    for cls, box in zip(
        ocr[0].boxes.cls.cpu().numpy(),
        ocr[0].boxes.xyxy.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        ch = plate_ocr_model.names[int(cls)]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        chars.append({"ch": ch, "cx": cx, "cy": cy})

    if len(chars) < 4:
        return None, None

    logger.info(f"[OCR] Detected {len(chars)} chars")

    if plate_type == "bsd":
        # Biển dài: 1 hàng, sort theo cx
        chars.sort(key=lambda c: c["cx"])
        layout = [c["ch"] for c in chars]
        return layout, chars
    else:
        # Biển vuông: 2 hàng
        # LOGIC: Gap-based cho biển thẳng, Median cho biển nghiêng
        
        # Sort theo cy để tìm gap
        chars_sorted_by_cy = sorted(chars, key=lambda c: c["cy"])
        cy_vals = [c["cy"] for c in chars_sorted_by_cy]
        
        # Tìm gap lớn nhất giữa các cy
        gaps = []
        for i in range(len(cy_vals) - 1):
            gap = cy_vals[i + 1] - cy_vals[i]
            gaps.append((i, gap))
        
        # Gap lớn nhất = ranh giới giữa 2 hàng
        max_gap_idx, max_gap = max(gaps, key=lambda x: x[1])
        threshold = (cy_vals[max_gap_idx] + cy_vals[max_gap_idx + 1]) / 2
        
        logger.info(f"[OCR] Gap-based threshold: {threshold:.1f} (max_gap={max_gap:.1f})")
        
        # Phân hàng sơ bộ
        upper = [c for c in chars if c["cy"] < threshold]
        lower = [c for c in chars if c["cy"] >= threshold]
        
        # PHÁT HIỆN NGHIÊNG: Tính slope của mỗi hàng
        def calculate_slope(row):
            """Tính độ nghiêng: cy thay đổi như thế nào theo cx"""
            if len(row) < 2:
                return 0
            row_sorted = sorted(row, key=lambda c: c["cx"])
            # Slope = (cy_cuối - cy_đầu) / (cx_cuối - cx_đầu)
            dx = row_sorted[-1]["cx"] - row_sorted[0]["cx"]
            dy = row_sorted[-1]["cy"] - row_sorted[0]["cy"]
            return dy / dx if dx != 0 else 0
        
        slope_upper = calculate_slope(upper) if len(upper) >= 2 else 0
        slope_lower = calculate_slope(lower) if len(lower) >= 2 else 0
        avg_slope = (slope_upper + slope_lower) / 2
        
        logger.info(f"[OCR] Slope: upper={slope_upper:.3f}, lower={slope_lower:.3f}, avg={avg_slope:.3f}")
        
        # Phân loại biển
        if abs(avg_slope) < 0.15:
            # Biển thẳng hoặc nghiêng nhẹ - GIỮ NGUYÊN gap-based (đang chạy tốt)
            logger.info(f"[OCR] Plate type: STRAIGHT (using gap-based)")
        else:
            # Biển nghiêng nặng - DÙNG MEDIAN (ổn định hơn cho demo)
            logger.info(f"[OCR] Plate type: TILTED (using median)")
            mid = np.median(cy_vals)
            upper = [c for c in chars if c["cy"] <= mid]
            lower = [c for c in chars if c["cy"] > mid]
            logger.info(f"[OCR] Switched to median threshold: {mid:.1f}")
        
        # Sort theo cx trong mỗi hàng
        upper.sort(key=lambda c: c["cx"])
        lower.sort(key=lambda c: c["cx"])
        
        logger.info(f"[OCR] Final - Upper: {[c['ch'] for c in upper]}, Lower: {[c['ch'] for c in lower]}")
        
        layout = [c["ch"] for c in upper + lower]
        return layout, chars


def build_plate_from_layout(layout, plate_memory):
    """
    Ghép biển số từ:
    - layout: thứ tự ký tự chuẩn (chuỗi phẳng)
    - plate_memory: OCR đa frame (list[str])
    """
    if not layout or not plate_memory:
        return "UNKNOWN"

    L = len(layout)

    # bucket[i] = danh sách ký tự OCR được ở vị trí i
    buckets = [[] for _ in range(L)]

    for text in plate_memory:
        if not text:
            continue
        text = normalize_text(text)

        # Chỉ xét chuỗi đủ dài
        if len(text) < L:
            continue

        for i in range(L):
            buckets[i].append(text[i])

    result = []

    for i, chars in enumerate(buckets):
        if not chars:
            # fallback: dùng layout nếu không có dữ liệu
            result.append(layout[i])
            continue

        counter = Counter(chars)
        best, count = counter.most_common(1)[0]

        # Heuristic sửa lỗi OCR phổ biến
        if best == "I" and counter.get("1", 0) >= count:
            best = "1"
        elif best == "O" and counter.get("0", 0) >= count:
            best = "0"

        result.append(best)

    return "".join(result)



# Video utility - Convert to web-playable MP4
def convert_to_web_mp4(input_path, output_path):
    """
    Convert video sang MP4 web-playable bằng ffmpeg
    - H.264
    - yuv420p
    - faststart
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg convert failed:\n{result.stderr.decode(errors='ignore')}"
        )

    if not os.path.exists(output_path):
        raise RuntimeError("FFmpeg convert failed: output file not created")

# Vietnamese plate utilities
VN_REGEX = re.compile(r'^(\d{2})([A-Z]{1,2})(\d{4,5})$')

def normalize_text(t):
    return re.sub(r'[^A-Z0-9]', '', t.upper())

def stabilize_numeric_tail(s):
    """
    Ổn định thứ tự chuỗi số: tránh bị xoay vòng do lệch bbox
    Áp dụng cho chuỗi 4–5 chữ số
    """
    if not s.isdigit() or len(s) < 4:
        return s

    # heuristic: nếu có đúng 1 số '0'
    if s.count("0") == 1:
        idx = s.index("0")

        # 0 thường đứng giữa chuỗi (vị trí 1 hoặc 2)
        if idx == 0:
            s = s[1:] + s[0]
        elif idx == len(s) - 1:
            s = s[-1] + s[:-1]

    return s

def format_vn_plate(s: str) -> str:
    """
    Format đơn giản cho biển VN (nếu muốn format kiểu 43F1-61261)
    Hàm này được gọi từ finalize_plate, giờ chỉ return s
    Format thực sự sẽ làm ở format_plate_with_type
    """
    return s

def validate_and_fix_plate_layout(upper, lower, plate_type="bsv"):
    """
    Validate và sửa layout biển số VN - PHIÊN BẢN MẠNH MẼ + TREND
    
    Dùng pattern matching + trend analysis để fix ký tự nhảy hàng
    
    Pattern biển VN:
    - Hàng trên: 2 số + 1-2 chữ + 0-1 ký tự (tổng 3-5 ký tự)
    - Hàng dưới: 4-5 số (toàn số)
    
    Trend analysis:
    - Biển nghiêng: cy có xu hướng tăng/giảm đều
    - Nếu có ký tự vi phạm trend → chuyển hàng
    
    Args:
        upper: list ký tự hàng trên
        lower: list ký tự hàng dưới
        plate_type: loại biển
    
    Returns:
        (upper_fixed, lower_fixed)
    """
    if plate_type != "bsv":
        return upper, lower
    
    # Gộp tất cả và sort theo cx
    all_chars = upper + lower
    all_chars.sort(key=lambda c: c["cx"])
    
    if len(all_chars) < 7:
        return upper, lower
    
    # BƯỚC 1: Tìm 2 số đầu tiên
    first_two_digits = []
    for c in all_chars:
        if c["ch"].isdigit():
            first_two_digits.append(c)
            if len(first_two_digits) == 2:
                break
    
    if len(first_two_digits) != 2:
        return upper, lower
    
    # BƯỚC 2: Tìm chữ cái sau 2 số (1-2 chữ)
    start_idx = all_chars.index(first_two_digits[1]) + 1
    letters = []
    
    for i in range(start_idx, min(start_idx + 2, len(all_chars))):
        if all_chars[i]["ch"].isalpha():
            letters.append(all_chars[i])
        else:
            break
    
    if len(letters) == 0:
        return upper, lower
    
    # BƯỚC 3: Kiểm tra ký tự sau chữ cái
    last_letter_idx = all_chars.index(letters[-1])
    
    if len(letters) == 2:
        # 2 chữ → không có gì sau
        upper_new = first_two_digits + letters
    else:
        # 1 chữ → có thể có 1 ký tự sau
        if last_letter_idx + 1 < len(all_chars):
            next_char = all_chars[last_letter_idx + 1]
            
            if next_char["ch"].isalnum():
                gap = next_char["cx"] - letters[-1]["cx"]
                
                gaps_so_far = []
                for i in range(len(first_two_digits) + len(letters) - 1):
                    g = all_chars[i + 1]["cx"] - all_chars[i]["cx"]
                    gaps_so_far.append(g)
                
                avg_gap = sum(gaps_so_far) / len(gaps_so_far) if gaps_so_far else 0
                
                if gap < avg_gap * 1.8:
                    upper_new = first_two_digits + letters + [next_char]
                else:
                    upper_new = first_two_digits + letters
            else:
                upper_new = first_two_digits + letters
        else:
            upper_new = first_two_digits + letters
    
    # BƯỚC 4: Phần còn lại là hàng dưới
    lower_new = [c for c in all_chars if c not in upper_new]
    
    # BƯỚC 5: KIỂM TRA CY ANOMALY - CHỈ KHI BIỂN CHƯA ĐƯỢC XOAY TỐT
    # Nếu biển đã xoay tốt, cy của các ký tự trong cùng hàng phải gần nhau
    # Chỉ fix khi có ký tự lệch quá nhiều (> 15 pixels)
    
    if len(lower_new) >= 3:
        # Tính độ lệch chuẩn của cy trong hàng dưới
        cy_values = [c["cy"] for c in lower_new]
        cy_mean = sum(cy_values) / len(cy_values)
        cy_std = (sum((cy - cy_mean) ** 2 for cy in cy_values) / len(cy_values)) ** 0.5
        
        # Chỉ fix nếu độ lệch chuẩn lớn (> 10) - nghĩa là biển vẫn còn nghiêng
        if cy_std > 10:
            cy_middle = [c["cy"] for c in lower_new[1:-1]]
            if cy_middle:
                cy_avg_middle = sum(cy_middle) / len(cy_middle)
                
                first_lower_cy = lower_new[0]["cy"]
                last_lower_cy = lower_new[-1]["cy"]
                
                # Ký tự đầu có cy THẤP hơn trung bình nhiều (> 15 pixels)
                if cy_avg_middle - first_lower_cy > 15:
                    char_to_move = lower_new.pop(0)
                    lower_new.append(char_to_move)
                    logger.info(f"[VALIDATE] Moved '{char_to_move['ch']}' from start to end (cy too low, std={cy_std:.1f})")
                
                # Ký tự cuối có cy THẤP hơn trung bình nhiều (> 15 pixels)
                elif cy_avg_middle - last_lower_cy > 15:
                    char_to_move = lower_new.pop(-1)
                    lower_new.insert(0, char_to_move)
                    logger.info(f"[VALIDATE] Moved '{char_to_move['ch']}' from end to start (cy too low, std={cy_std:.1f})")
        else:
            logger.info(f"[VALIDATE] Plate well-aligned (cy_std={cy_std:.1f}), skip cy anomaly fix")
    
    # BƯỚC 6: Validate kết quả
    if len(upper_new) not in [3, 4, 5] or len(lower_new) not in [4, 5]:
        logger.warning(f"Pattern validation failed: upper={len(upper_new)}, lower={len(lower_new)}")
        return upper, lower
    
    if not all(c["ch"].isdigit() for c in lower_new):
        logger.warning(f"Lower row contains non-digits")
        return upper, lower
    
    if not (upper_new[0]["ch"].isdigit() and upper_new[1]["ch"].isdigit()):
        logger.warning(f"Upper row doesn't start with 2 digits")
        return upper, lower
    
    return upper_new, lower_new

def format_plate_with_type(plate_text, plate_type, chars_data=None):
    """
    Format biển số VN theo chuẩn
    
    Biển vuông (BSV):
    - 43A 123.45 (1 chữ, không có số sau)
    - 43-A1 123.45 (1 chữ + 1 số)
    - 43-AA 123.45 (2 chữ, KHÔNG BAO GIỜ có số sau)
    
    Biển dài (BSD):
    - 43H-191.45 (2 số + 1 chữ + 5 số, format: XXY-ZZZ.WW)
    
    Args:
        plate_text: chuỗi biển số đã normalize (vd: 43F161261)
        plate_type: "bsv" (vuông) hoặc "bsd" (dài)
        chars_data: list of dict với cx, cy, ch
    
    Returns:
        Biển số đã format
    """
    if not plate_text or plate_text in ["UNKNOWN", "Không đọc được"]:
        return plate_text
    
    # Normalize
    plate_text = normalize_text(plate_text)
    
    if plate_type == "bsd":
        # Biển dài: 2 số + 1 chữ + 5 số
        # Format: 43H19145 -> 43H-191.45
        match = re.match(r'^(\d{2})([A-Z])(\d{5})$', plate_text)
        if match:
            province = match.group(1)  # 43
            letter = match.group(2)     # H
            numbers = match.group(3)    # 19145
            return f"{province}{letter}-{numbers[:3]}.{numbers[3:]}"
        else:
            # Không match format chuẩn, trả về nguyên bản
            return plate_text
    else:
        # Biển vuông (BSV)
        # Format: 43F161261, 43AB161261, 43F61261, 43AB1234, etc.
        match = re.match(r'^(\d{2})([A-Z]{1,2})([A-Z0-9]?)(\d{4,5})$', plate_text)
        
        if not match:
            return plate_text
        
        province = match.group(1)  # 43
        letters = match.group(2)   # F hoặc AB
        mid_char = match.group(3)  # 1 hoặc rỗng
        tail = match.group(4)       # 61261 hoặc 1234
        
        # Format phần đầu
        if len(letters) == 2:
            # 2 chữ → luôn có dấu gạch, bỏ mid_char (nếu có)
            front = f"{province}-{letters}"
        elif mid_char:
            # 1 chữ + số → có dấu gạch
            front = f"{province}-{letters}{mid_char}"
        else:
            # 1 chữ, không có số → không có dấu gạch
            front = f"{province}{letters}"
        
        # Format phần số cuối
        if len(tail) == 5:
            back = f"{tail[:3]}.{tail[3:]}"
        else:
            back = tail
        
        return f"{front} {back}"

def finalize_plate(texts):
    """
    Chuẩn hóa và ổn định biển số VN
    - Hỗ trợ 1 hoặc 2 chữ cái
    - Ổn định thứ tự đuôi số
    - Không sửa bừa nếu không đúng format
    """
    if not texts:
        return "UNKNOWN"

    norm = [normalize_text(t) for t in texts if t]

    VN_FULL_REGEX = re.compile(r'^(\d{2})([A-Z]{1,2})(\d)(\d{4,5})$')
    valid = [t for t in norm if VN_FULL_REGEX.match(t)]

    if valid:
        best = Counter(valid).most_common(1)[0][0]
    else:
        best = Counter(norm).most_common(1)[0][0]

    m = VN_FULL_REGEX.match(best)
    if not m:
        return best if best else "UNKNOWN"

    province, letters, mid_digit, tail = m.groups()
    tail = stabilize_numeric_tail(tail)
    best = f"{province}{letters}{mid_digit}{tail}"

    return format_vn_plate(best)

# OCR functions
def ocr_plate(plate_img, fid, redis, plate_type="bsv"):
    """
    OCR biển số
    
    Args:
        plate_img: ảnh biển số
        fid: file id
        redis: redis connection
        plate_type: "bsv" (biển số vuông - 2 hàng) hoặc "bsd" (biển số dài - 1 hàng)
    """
    ocr = plate_ocr_model(plate_img, conf=0.3)
    if not ocr or ocr[0].boxes is None:
        return []

    chars = []
    for cls, box in zip(
        ocr[0].boxes.cls.cpu().numpy(),
        ocr[0].boxes.xyxy.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        ch = plate_ocr_model.names[int(cls)]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        chars.append({"cx": cx, "cy": cy, "ch": ch})

    if len(chars) < 4:
        return []

    # Xử lý theo loại biển
    if plate_type == "bsd":
        # Biển số dài: 1 hàng duy nhất - CHỈ sort theo cx
        chars.sort(key=lambda c: c["cx"])
        text = "".join(c["ch"] for c in chars)
    else:
        # Biển số vuông: 2 hàng - dùng hybrid approach
        cy_vals = [c["cy"] for c in chars]
        
        # Phương pháp 1: Median (ổn định)
        mid = np.median(cy_vals)
        upper_median = [c for c in chars if c["cy"] <= mid]
        lower_median = [c for c in chars if c["cy"] > mid]
        
        # Validate median
        median_valid = (len(upper_median) in [3, 4, 5] and len(lower_median) in [4, 5])
        
        if median_valid:
            # Median OK, dùng luôn
            upper = upper_median
            lower = lower_median
        else:
            # Median không OK, thử gap-based
            cy_sorted = sorted(cy_vals)
            
            if len(cy_sorted) > 1:
                gaps = [cy_sorted[i+1] - cy_sorted[i] for i in range(len(cy_sorted)-1)]
                max_gap = max(gaps)
                
                # Chỉ dùng gap nếu đủ lớn (> 5 pixels)
                if max_gap > 5:
                    max_gap_idx = gaps.index(max_gap)
                    threshold = (cy_sorted[max_gap_idx] + cy_sorted[max_gap_idx+1]) / 2
                    
                    upper = [c for c in chars if c["cy"] < threshold]
                    lower = [c for c in chars if c["cy"] >= threshold]
                else:
                    # Gap không đủ lớn, fallback về median
                    upper = upper_median
                    lower = lower_median
            else:
                # Không đủ dữ liệu, dùng median
                upper = upper_median
                lower = lower_median

        upper.sort(key=lambda c: c["cx"])
        lower.sort(key=lambda c: c["cx"])

        text = "".join(c["ch"] for c in upper + lower)

        upper.sort(key=lambda c: (c["cx"], c["cy"]))
        lower.sort(key=lambda c: (c["cx"], c["cy"]))

        text = "".join(c["ch"] for c in upper + lower)
    
    return [text]

# Image processing
def process_image(path, fid, redis):
    logger.info(f"Bắt đầu xử lý ảnh: {fid}")
    redis.set(fid, json.dumps({"status": "processing", "progress": 0}))

    try:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Không thể đọc ảnh: {path}")
            redis.set(fid, json.dumps({"status": "error", "progress": 0, "message": "Không thể đọc ảnh"}))
            return
        
        logger.debug(f"Kích thước ảnh: {img.shape}")
        vehicles = []

        # DETECT XE
        logger.debug("Bắt đầu detect phương tiện")
        rs = vehicle_model(img, conf=0.45, classes=[2, 3, 5, 7])

        if rs and rs[0].boxes is not None:
            boxes = rs[0].boxes.xyxy.cpu().numpy()
            logger.info(f"Đã detect được {len(boxes)} phương tiện (trước filter)")
            
            # Filter overlapping boxes để loại bỏ duplicate
            boxes = filter_overlapping_boxes(boxes, iou_threshold=0.4)
            logger.info(f"Còn lại {len(boxes)} phương tiện (sau filter)")

            # ƯU TIÊN XE LỚN NHẤT
            boxes = sorted(
                boxes,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                reverse=True
            )

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # LƯU ẢNH XE
                veh_name = f"{fid}_veh_{i}_output.jpg"
                cv2.imwrite(os.path.join(STATIC_IMG, veh_name), crop)

                # VẼ BBOX XE
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Xe {i+1}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                # DETECT BIỂN
                plate_img_name = ""
                plate_final = "Không đọc được"

                plates = plate_crop_model(crop, conf=0.4)
                if plates and plates[0].boxes is not None:
                    pboxes = plates[0].boxes.xyxy.cpu().numpy()
                    pclasses = plates[0].boxes.cls.cpu().numpy()

                    # lấy biển lớn nhất
                    plate_data = list(zip(pboxes, pclasses))
                    plate_data = sorted(
                        plate_data,
                        key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]),
                        reverse=True
                    )

                    for pbox, pcls in plate_data:
                        px1, py1, px2, py2 = map(int, pbox)
                        plate_crop = crop[py1:py2, px1:px2]
                        if plate_crop.size == 0:
                            continue

                        # Lưu ảnh biển số gốc
                        plate_img_name = f"{fid}_plate_{i}.jpg"
                        cv2.imwrite(os.path.join(STATIC_IMG, plate_img_name), plate_crop)
                        
                        # Lấy loại biển
                        plate_type = plate_crop_model.names[int(pcls)].lower()
                        logger.info(f"[IMAGE] Plate type from model: {plate_type}")
                        
                        # OCR trực tiếp trên ảnh gốc (không xoay)
                        layout, chars_data = ocr_plate_layout(plate_crop, plate_type)
                        
                        if layout:
                            plate_final = "".join(layout)
                        else:
                            plate_final = "Không đọc được"
                        
                        logger.info(f"[IMAGE] Plate: {plate_final}")
                        
                        # AUTO-DETECT: Nếu có 8 ký tự theo pattern BSD → override plate_type
                        if len(plate_final) == 8:
                            # Check pattern: 2 số + 1 chữ + 5 số
                            if (plate_final[0:2].isdigit() and 
                                plate_final[2].isalpha() and 
                                plate_final[3:8].isdigit()):
                                plate_type = "bsd"
                                logger.info(f"[IMAGE] Auto-corrected plate_type to BSD (pattern: 2digit+1letter+5digit)")
                        
                        # Format
                        if plate_final not in ["UNKNOWN", "Không đọc được"]:
                            plate_final = format_plate_with_type(plate_final, plate_type, chars_data)
                        
                        logger.info(f"[IMAGE] Formatted: {plate_final}")
                        
                        break

                vehicles.append({
                    "vehicle_index": i + 1,  # Thêm vehicle_index để hiển thị "Xe 1, Xe 2,..."
                    "vehicle_image": veh_name,
                    "plate_image": plate_img_name,
                    "plate_final": plate_final,
                    "bbox": [x1, y1, x2, y2]
                })

        # LƯU ẢNH TỔNG
        out_img = f"{fid}_output.jpg"
        cv2.imwrite(os.path.join(STATIC_IMG, out_img), img)

        meta = {"type": "image", "output": out_img, "vehicles": vehicles}

        with open(os.path.join(OUTPUT_DIR, fid + ".json"), "w", encoding="utf8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Hoàn thành xử lý ảnh: {fid}, tìm thấy {len(vehicles)} phương tiện")
        redis.set(fid, json.dumps({"status": "done", "progress": 100}))

    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh {fid}: {str(e)}", exc_info=True)
        redis.set(fid, json.dumps({"status": "error", "progress": 0, "message": str(e)}))

# Sharpness calculation
def calc_sharpness(img):
    if img is None or not hasattr(img, "size") or img.size == 0:
        return 0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except cv2.error:
        return 0

# Plate rotation utilities
def calculate_plate_angle_from_chars(chars):
    """
    Tính góc nghiêng của biển số dựa trên tọa độ ký tự
    
    STRATEGY MỚI: Phân 2 hàng trước (dùng K-means), rồi tính góc của TỪNG hàng
    
    Args:
        chars: list of dict với cx, cy
    
    Returns:
        angle: góc nghiêng (độ), dương = nghiêng phải, âm = nghiêng trái
    """
    if len(chars) < 4:
        return 0
    
    try:
        # Dùng K-means để phân 2 hàng
        from sklearn.cluster import KMeans
        points = np.array([[c["cx"], c["cy"]] for c in chars])
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(points)
        labels = kmeans.labels_
        
        # Phân chia theo cluster
        cluster_0 = [chars[i] for i in range(len(chars)) if labels[i] == 0]
        cluster_1 = [chars[i] for i in range(len(chars)) if labels[i] == 1]
        
        # Xác định cluster nào là hàng trên (cy nhỏ hơn)
        cy_avg_0 = sum(c["cy"] for c in cluster_0) / len(cluster_0)
        cy_avg_1 = sum(c["cy"] for c in cluster_1) / len(cluster_1)
        
        if cy_avg_0 < cy_avg_1:
            upper = cluster_0
            lower = cluster_1
        else:
            upper = cluster_1
            lower = cluster_0
        
        # Tính góc của hàng trên (đáng tin hơn vì có pattern cố định)
        if len(upper) >= 3:
            upper.sort(key=lambda c: c["cx"])
            
            # Lấy 2 điểm đầu và cuối
            first = upper[0]
            last = upper[-1]
            
            dx = last["cx"] - first["cx"]
            dy = last["cy"] - first["cy"]
            
            if dx == 0:
                return 0
            
            angle = np.degrees(np.arctan(dy / dx))
            
            logger.info(f"[CALC_ANGLE] Upper row: {len(upper)} chars, angle={angle:.2f}°")
            return angle
        else:
            return 0
            
    except Exception as e:
        logger.warning(f"[CALC_ANGLE] Failed: {e}")
        return 0

def rectify_plate_image_with_chars(plate_img, chars):
    """
    Xoay biển số dựa trên tọa độ ký tự OCR
    Chính xác hơn minAreaRect vì dựa trên nội dung thực
    
    Args:
        plate_img: ảnh biển số
        chars: list of dict với cx, cy từ OCR
    
    Returns:
        rotated image
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    if not chars or len(chars) < 3:
        return plate_img
    
    try:
        # Tính góc nghiêng từ ký tự
        angle = calculate_plate_angle_from_chars(chars)
        
        logger.info(f"[RECTIFY_CHARS] Calculated angle: {angle:.2f} degrees")
        
        # Chỉ xoay nếu góc > 1 độ
        if abs(angle) < 1:
            logger.info(f"[RECTIFY_CHARS] Angle too small, skip rotation")
            return plate_img
        
        # Giới hạn góc - TĂNG LÊN để xử lý biển nghiêng nặng
        if abs(angle) > 45:
            logger.warning(f"[RECTIFY_CHARS] Angle too large ({angle:.2f}), clamping to ±45")
            angle = 45 if angle > 0 else -45
        
        # Xoay ảnh
        (h, w) = plate_img.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # Âm để xoay ngược lại
        rotated = cv2.warpAffine(
            plate_img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.info(f"[RECTIFY_CHARS] Rotated plate by {-angle:.2f} degrees")
        return rotated
    
    except Exception as e:
        logger.warning(f"[RECTIFY_CHARS] Failed: {e}")
        return plate_img

def rectify_plate_image(plate_img):
    """
    Xoay biển số từ xiên về gần ngang nhất có thể
    Xoay mọi biển nghiêng để OCR chính xác hơn
    
    LƯU Ý: Hàm này dùng minAreaRect, không chính xác với biển crop chữ nhật
    Nên dùng rectify_plate_image_with_chars() nếu có tọa độ ký tự
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Tìm contour
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return plate_img

        # Lấy contour lớn nhất (giả định là biển)
        cnt = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (width, height), angle = rect

        # minAreaRect trả về góc từ -90 đến 0
        # Nếu width < height nghĩa là đo nhầm chiều -> swap và điều chỉnh góc
        if width < height:
            width, height = height, width
            angle = angle + 90

        # BỎ THRESHOLD - Xoay mọi biển nghiêng
        # Chỉ bỏ qua nếu góc quá nhỏ (< 1 độ)
        if abs(angle) < 1:
            return plate_img

        # Giới hạn góc xoay trong khoảng hợp lý
        if abs(angle) > 45:
            return plate_img

        # Xoay ảnh
        (h, w) = plate_img.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            plate_img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        logger.info(f"[RECTIFY] Rotated plate by {angle:.2f} degrees")
        return rotated

    except Exception as e:
        logger.warning(f"[RECTIFY] Failed to rectify: {e}")
        return plate_img

# Video processing
def process_video(path, fid, redis):
    """
    Xử lý video - LOGIC ĐƠN GIẢN:
    1. Track phương tiện
    2. OCR biển số mỗi frame → lưu vào plate_memory
    3. Tìm consensus (mẫu số chung) từ plate_memory
    4. Tìm frame có biển số = consensus VÀ rõ nhất (area lớn nhất)
    5. Dùng frame đó để format biển số cuối cùng
    6. Smooth bbox để vẽ mượt hơn
    """
    logger.info(f"Bắt đầu xử lý video: {fid}")
    redis.set(fid, json.dumps({"status": "processing", "progress": 0}))

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        redis.set(fid, json.dumps({
            "status": "error",
            "progress": 0,
            "message": "Không mở được video"
        }))
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    raw_path = os.path.join(STATIC_VIDEO, f"{fid}_raw.avi")
    final_path = os.path.join(STATIC_VIDEO, f"{fid}.mp4")

    out = cv2.VideoWriter(
        raw_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (w, h)
    )

    tracks = {}
    track_id_to_vehicle_idx = {}
    next_vehicle_idx = 1
    frame_id = 0

    redis.set(fid, json.dumps({
        "status": "processing",
        "progress": 5,
        "message": "Đang detect & tracking phương tiện..."
    }))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rs = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.45,
            classes=[2, 3, 5, 7]
        )

        if rs and rs[0].boxes is not None and rs[0].boxes.id is not None:
            boxes = rs[0].boxes.xyxy.cpu().numpy()
            ids = rs[0].boxes.id.cpu().numpy()

            for box, tid in zip(boxes, ids):
                track_id = int(tid)
                x1, y1, x2, y2 = map(int, box)

                if track_id not in track_id_to_vehicle_idx:
                    track_id_to_vehicle_idx[track_id] = next_vehicle_idx
                    next_vehicle_idx += 1

                if track_id not in tracks:
                    tracks[track_id] = {
                        "frames_seen": 0,
                        "best_crop": None,
                        "best_area": 0,
                        "plate_memory": [],  # Lưu tất cả OCR results
                        "plate_frames": [],  # Lưu (plate_text, plate_img, area, plate_type)
                        "last_bbox": None    # Bbox frame trước để smooth
                    }

                track = tracks[track_id]
                track["frames_seen"] += 1

                # Smooth bbox: nếu bbox mới không lệch quá nhiều so với bbox cũ
                if track["last_bbox"] is not None:
                    lx1, ly1, lx2, ly2 = track["last_bbox"]
                    # Tính độ lệch
                    dx = abs((x1 + x2) / 2 - (lx1 + lx2) / 2)
                    dy = abs((y1 + y2) / 2 - (ly1 + ly2) / 2)
                    
                    # Nếu lệch quá 50 pixels → giữ bbox cũ (smooth)
                    if dx < 50 and dy < 50:
                        # Smooth: trung bình giữa bbox cũ và mới
                        x1 = int(0.7 * lx1 + 0.3 * x1)
                        y1 = int(0.7 * ly1 + 0.3 * y1)
                        x2 = int(0.7 * lx2 + 0.3 * x2)
                        y2 = int(0.7 * ly2 + 0.3 * y2)
                
                track["last_bbox"] = (x1, y1, x2, y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area > track["best_area"]:
                    track["best_area"] = area
                    track["best_crop"] = crop.copy()

                # Detect biển số
                plates = plate_crop_model(crop, conf=0.4)
                if plates and plates[0].boxes is not None:
                    pboxes = plates[0].boxes.xyxy.cpu().numpy()
                    pclasses = plates[0].boxes.cls.cpu().numpy()
                    
                    for pbox, pcls in zip(pboxes, pclasses):
                        px1, py1, px2, py2 = map(int, pbox)
                        plate_crop = crop[py1:py2, px1:px2]
                        if plate_crop.size == 0:
                            continue

                        plate_type = plate_crop_model.names[int(pcls)].lower()
                        p_area = (px2 - px1) * (py2 - py1)
                        
                        # OCR đơn giản - KHÔNG xoay
                        layout, chars_data = ocr_plate_layout(plate_crop, plate_type)
                        
                        if layout:
                            plate_text = "".join(layout)
                            
                            # Auto-detect BSD
                            if len(plate_text) == 8:
                                if (plate_text[0:2].isdigit() and 
                                    plate_text[2].isalpha() and 
                                    plate_text[3:8].isdigit()):
                                    plate_type = "bsd"
                            
                            # Lưu vào memory
                            track["plate_memory"].append(plate_text)
                            
                            # Lưu frame này (để sau tìm frame đẹp nhất)
                            track["plate_frames"].append({
                                "text": plate_text,
                                "image": plate_crop.copy(),
                                "area": p_area,
                                "type": plate_type,
                                "chars_data": chars_data
                            })
                        
                        break

                # Vẽ bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        frame_id += 1

        if frame_id % 10 == 0:
            redis.set(fid, json.dumps({
                "status": "processing",
                "progress": int(frame_id / total * 60),
                "message": f"Đang xử lý frame {frame_id}/{total}"
            }))

    cap.release()
    out.release()

    redis.set(fid, json.dumps({
        "status": "processing",
        "progress": 80,
        "message": "Đang convert video..."
    }))

    convert_to_web_mp4(raw_path, final_path)

    if not os.path.exists(final_path):
        raise RuntimeError("Convert video thất bại")

    if os.path.exists(raw_path):
        os.remove(raw_path)

    # XỬ LÝ KẾT QUẢ
    vehicles = []

    for track_id, track in tracks.items():
        if track["frames_seen"] < 10 or track["best_crop"] is None:
            continue

        veh_img = f"{fid}_veh_{track_id}.jpg"
        cv2.imwrite(os.path.join(STATIC_IMG, veh_img), track["best_crop"])

        plate_img = ""
        plate_final = "Không đọc được"

        if len(track["plate_memory"]) > 0:
            # Tìm consensus (mẫu số chung)
            from collections import Counter
            counter = Counter(track["plate_memory"])
            consensus, count = counter.most_common(1)[0]
            
            logger.info(f"[VIDEO] Track {track_id}: Consensus={consensus} (count={count}/{len(track['plate_memory'])})")
            
            # Tìm frame có plate_text = consensus VÀ area lớn nhất
            best_frame = None
            best_area = 0
            
            for pf in track["plate_frames"]:
                if pf["text"] == consensus and pf["area"] > best_area:
                    best_frame = pf
                    best_area = pf["area"]
            
            if best_frame:
                # Lưu ảnh biển số từ frame đẹp nhất
                plate_img = f"{fid}_plate_{track_id}.jpg"
                cv2.imwrite(os.path.join(STATIC_IMG, plate_img), best_frame["image"])
                
                plate_type = best_frame["type"]
                chars_data = best_frame["chars_data"]
                
                logger.info(f"[VIDEO] Track {track_id}: Best frame - text={best_frame['text']}, area={best_area}, type={plate_type}")
                
                # Format biển số
                plate_final = format_plate_with_type(consensus, plate_type, chars_data)
                logger.info(f"[VIDEO] Track {track_id}: Formatted={plate_final}")
            else:
                plate_final = consensus

        vehicles.append({
            "vehicle_index": track_id_to_vehicle_idx.get(track_id, track_id),
            "track_id": track_id,
            "vehicle_image": veh_img,
            "plate_image": plate_img,
            "plate_final": plate_final,
            "frames_seen": track["frames_seen"]
        })

    vehicles = merge_tracks_by_plate(vehicles)

    with open(os.path.join(OUTPUT_DIR, fid + ".json"), "w", encoding="utf8") as f:
        json.dump({
            "type": "video",
            "output": f"{fid}.mp4",
            "vehicles": vehicles
        }, f, ensure_ascii=False, indent=2)

    redis.set(fid, json.dumps({
        "status": "done",
        "progress": 100
    }))

    logger.info(f"Hoàn thành video {fid}")
