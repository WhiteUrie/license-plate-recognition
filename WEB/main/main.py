from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
import threading
import json
import logging
from datetime import datetime
from processor import process_file
from redis import Redis

# App configuration
REDIS_TASK_TTL = 60 * 60  # 1 hour

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Giới hạn 500MB
redis = Redis(host='localhost', port=6379, db=0)  # Kết nối với Redis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Logging configuration
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'app.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Validation configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'}

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_IMG = os.path.join(BASE_DIR, "static", "images")
STATIC_VIDEO = os.path.join(BASE_DIR, "static", "videos")

for p in [UPLOAD_DIR, OUTPUT_DIR, STATIC_IMG, STATIC_VIDEO]:
    os.makedirs(p, exist_ok=True)

# Routes
@app.route("/")
def index():
    logger.info("Trang chủ được truy cập")
    files = load_processed_files()
    logger.info(f"Đã load {len(files)} file đã xử lý")
    return render_template("index.html", files=files)

@app.route("/upload", methods=["POST"])
def upload():
    save_path = None
    fid = None

    try:
        logger.info("📥 Nhận request upload")

        # Read form data
        file = request.files.get("file")
        cam = request.form.get("cam", "").strip()
        road = request.form.get("road", "").strip()
        date = request.form.get("date", "").strip()
        time_ = request.form.get("time", "").strip()

        logger.debug(f"Form: cam={cam}, road={road}, date={date}, time={time_}")

        # Basic validation
        if not file or file.filename == "":
            logger.warning("❌ Upload: chưa chọn file")
            return jsonify({"error": "Chưa chọn file"}), 400

        # Auto-fill cam and road if empty
        if not cam:
            cam = "cam1"
            logger.info(f"📷 Auto-filled cam: {cam}")
        if not road:
            road = "noname"
            logger.info(f"🛣️ Auto-filled road: {road}")

        # Auto-fill date/time if empty
        if not date or not time_:
            now = datetime.now()
            date = date or now.strftime("%Y%m%d")
            time_ = time_ or now.strftime("%H%M%S")

        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower()
        logger.debug(f"File extension: {ext}")
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"❌ Upload: định dạng không hỗ trợ: {ext}")
            return jsonify({
                "error": f"Định dạng file không hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Validate file size
        if request.content_length and request.content_length > MAX_FILE_SIZE:
            size_mb = request.content_length / (1024 * 1024)
            logger.warning(f"❌ Upload: file quá lớn (content_length={size_mb:.2f}MB)")
            return jsonify({
                "error": f"File quá lớn. Tối đa: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            }), 400

        # Build file ID and path
        fid = build_file_id(cam, road, date, time_)
        filename = secure_filename(fid + ext)
        save_path = os.path.join(UPLOAD_DIR, filename)

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Check for duplicate ID
        if redis.exists(fid):
            logger.warning(f"⚠️ Upload trùng ID: {fid}")
            return jsonify({
                "error": "File/job với ID này đã tồn tại. Hãy đổi date/time hoặc thử lại sau.",
                "id": fid
            }), 409

        # Create Redis task
        task_info = {
            "status": "uploading",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "filename": filename,
            "cam": cam,
            "road": road
        }

        redis.setex(fid, REDIS_TASK_TTL, json.dumps(task_info))
        logger.info(f"✅ Redis created task: {fid} (TTL={REDIS_TASK_TTL}s)")

        # Save file
        file.save(save_path)
        logger.info(f"✅ File saved: {save_path}")

        # Verify file size after save
        actual_size = os.path.getsize(save_path)
        logger.debug(f"Actual size: {actual_size / (1024*1024):.2f}MB")

        if actual_size > MAX_FILE_SIZE:
            os.remove(save_path)
            redis.setex(fid, REDIS_TASK_TTL, json.dumps({
                "status": "failed",
                "progress": 0,
                "error": "File quá lớn sau khi lưu"
            }))
            logger.warning(f"❌ File quá lớn sau lưu => xóa: {save_path}")
            return jsonify({
                "error": f"File quá lớn. Tối đa: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
            }), 400

        # Update Redis status
        redis.setex(fid, REDIS_TASK_TTL, json.dumps({
            "status": "processing",
            "progress": 0,
            "created_at": task_info["created_at"],
            "filename": filename,
            "cam": cam,
            "road": road
        }))

        # Start background worker
        def _worker():
            try:
                logger.info(f"⚙️ Worker start processing: {fid}")
                process_file(save_path, fid, redis)
                logger.info(f"✅ Worker done: {fid}")

            except Exception as e:
                logger.error(f"❌ Worker error: {fid} - {e}", exc_info=True)
                redis.setex(fid, REDIS_TASK_TTL, json.dumps({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e)
                }))

                # Cleanup file on error
                try:
                    if save_path and os.path.exists(save_path):
                        os.remove(save_path)
                        logger.warning(f"🧹 Cleanup file on worker error: {save_path}")
                except Exception:
                    logger.warning("Cleanup failed", exc_info=True)

        threading.Thread(target=_worker, daemon=True).start()

        # Return response immediately
        logger.info(f"🚀 Upload OK: {fid}")
        return jsonify({"id": fid})

    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)

        # Cleanup on error
        try:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
                logger.warning(f"🧹 Cleanup file because upload failed: {save_path}")
        except Exception:
            logger.warning("Cleanup failed", exc_info=True)

        # Update Redis on error
        try:
            if fid:
                redis.setex(fid, REDIS_TASK_TTL, json.dumps({
                    "status": "failed",
                    "progress": 0,
                    "error": str(e)
                }))
        except Exception:
            pass

        return jsonify({"error": str(e)}), 500

@app.route("/status/<fid>")
def get_status(fid):
    status_data = redis.get(fid)
    if not status_data:
        return jsonify({"error": "Không tìm thấy trạng thái"}), 404
    return jsonify(json.loads(status_data))

@app.route("/results/<fid>")
def result(fid):
    meta_path = os.path.join(OUTPUT_DIR, fid + ".json")
    if not os.path.exists(meta_path):
        return "Không tìm thấy kết quả", 404

    with open(meta_path, encoding="utf8") as f:
        meta = json.load(f)

    return render_template("result.html", fid=fid, meta=meta)

@app.route("/static/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(STATIC_IMG, filename)

@app.route("/static/videos/<path:filename>")
def serve_videos(filename):
    return send_from_directory(STATIC_VIDEO, filename)

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        "error": f"File quá lớn. Kích thước tối đa: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
    }), 413

# Utility functions
def build_file_id(cam, road, date, time):
    road = road.replace(" ", "")
    return f"{cam}_{road}_{date}_{time}"

def parse_file_id(fid):
    try:
        cam, road, date, time = fid.split("_", 3)
        return {
            "cam": cam,
            "road": road,
            "date": date,
            "time": time
        }
    except:
        return {
            "cam": "?",
            "road": "?",
            "date": "?",
            "time": "?"
        }

def parse_filename(filename):
    """
    Parse tên file để lấy thông tin cam, road, date, time
    Format: cam1_nguyenhuutho_20260403_070844.mp4
    Trả về dict với các key: cam, road, date, time hoặc None nếu không parse được
    """
    # Bỏ đuôi file
    name_without_ext = os.path.splitext(filename)[0]
    
    # Thử parse format: cam_road_date_time
    parts = name_without_ext.split("_")
    
    if len(parts) >= 4:
        # Kiểm tra date có đúng format YYYYMMDD (8 chữ số) không
        date_part = parts[-2]
        # Kiểm tra time có đúng format HHMMSS (6 chữ số) không
        time_part = parts[-1]
        
        if len(date_part) == 8 and date_part.isdigit() and len(time_part) == 6 and time_part.isdigit():
            cam = parts[0]
            road = "_".join(parts[1:-2])  # Road có thể có dấu gạch dưới
            date = date_part
            time = time_part
            
            return {
                "cam": cam,
                "road": road,
                "date": date,
                "time": time
            }
    
    return None

def load_processed_files():
    """
    Load danh sách các file đã xử lý và sắp xếp theo thời gian xử lý (mới nhất lên đầu)
    """
    files = []
    for f in os.listdir(OUTPUT_DIR):
        if not f.endswith(".json"):
            continue
        fid = f.replace(".json", "")
        info = parse_file_id(fid)
        meta_path = os.path.join(OUTPUT_DIR, f)
        
        # Lấy thời gian modification của file (thời điểm file được tạo/sửa)
        file_mtime = os.path.getmtime(meta_path)
        
        with open(meta_path, encoding="utf8") as jf:
            meta = json.load(jf)

        files.append({
            "id": fid,
            "type": meta.get("type"),
            "mtime": file_mtime,  # Thời gian modification để sắp xếp
            **info
        })
    
    # Sắp xếp theo thời gian modification (mới nhất lên đầu)
    return sorted(files, key=lambda x: x["mtime"], reverse=True)

# Main entry point
if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)
