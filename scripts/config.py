import os
import json
import logging
import warnings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")


class Paths:
    """Quản lý toàn bộ đường dẫn và ánh xạ model theo JSON cấu hình."""

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    MAP_FILE = os.path.join(ROOT_DIR, "models_map.json")

    # Các key tương ứng với loại model
    MODEL_KEYS = [
        "WORD2VEC",
        "CHAR_TOKENIZER",
        "XGB",
        "RF",
        "CNN",
        "CNN_LSTM",
        "CHAR_CNN",
        "CHAR_CNN_LSTM",
        "CNN_HYBRID",
        "ALBERT",
        "MOBILE_BERT",
    ]

    # Nạp JSON map khi khởi tạo
    def __init__(self):
        self.model_map = self._load_model_map()
        self.paths = self._build_paths_from_map()

    def _load_model_map(self):
        """Đọc file JSON map: tên model PATH → {file_id, filename, is_zip}."""
        if not os.path.exists(self.MAP_FILE):
            raise FileNotFoundError(f"❌ Model map JSON not found: {self.MAP_FILE}")
        with open(self.MAP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_paths_from_map(self):
        """Tạo dict ánh xạ model_key → thông tin đầy đủ (path, filename, file_id, is_zip)."""
        paths = {}

        for key in self.MODEL_KEYS:
            base_path = getattr(self, key, self.MODELS_DIR)
            info = self.model_map.get(key)

            if info:
                filename = info.get("filename", "")
                full_path = os.path.join(base_path, filename)
                paths[key] = {
                    "full_path": full_path,
                    "base_path": base_path,
                    "filename": filename,
                    "file_id": info.get("file_id"),
                    "is_zip": info.get("is_zip", False),
                }
            else:
                # Không có trong JSON → chỉ lưu base path
                paths[key] = {
                    "full_path": base_path,
                    "base_path": base_path,
                    "filename": None,
                    "file_id": None,
                    "is_zip": None,
                }
                logging.warning(f"⚠️  No entry found for {key} in {self.MAP_FILE}")

        return paths
