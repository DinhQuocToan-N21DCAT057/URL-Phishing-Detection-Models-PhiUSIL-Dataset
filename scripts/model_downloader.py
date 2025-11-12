from config import Paths
import os, gdown, zipfile
import logging
import warnings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")


class ModelDownloader:
    def __init__(self):
        self.cfg = Paths()

    def _download_from_drive(self, file_id, dest_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logging.info(f"⬇️ Downloading {dest_path}")
        try:
            gdown.download(url, dest_path, quiet=False)
        except gdown.exceptions.FileURLRetrievalError as e:
            logging.info(f"⬇❌ Error downloading {url}! Exception: {e}")
            return

    def _extract_zip(self, zip_path, extract_to):
        logging.info(f"📦 Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(zip_path)

    def check_and_download_all(self):
        """Kiểm tra toàn bộ model, nếu thiếu thì tải."""
        for key, info in self.cfg.paths.items():
            self._check_and_download_single(key, info)
        logging.info("🎯 All model checks complete.")

    def _check_and_download_single(self, key, info):
        """Hàm nội bộ để kiểm tra và tải 1 model."""
        file_id = info["file_id"]
        full_path = info["full_path"]
        is_zip = info["is_zip"]

        if not file_id:
            logging.warning(f"⚠️ No Google Drive ID for {key}")
            return

        if os.path.exists(full_path):
            logging.info(f"✅ Found: {key}")
            return

        logging.warning(f"🚨 Missing model: {key} → {full_path}")

        if is_zip:
            zip_temp = os.path.join(self.cfg.ROOT_DIR, f"{key}_temp.zip")
            self._download_from_drive(file_id, zip_temp)
            self._extract_zip(zip_temp, os.path.dirname(full_path))
        else:
            self._download_from_drive(file_id, full_path)

    def download_single(self, model_key: str, force=False):
        """
        Tải riêng 1 model (dùng để debug hoặc kiểm thử).
        - model_key: tên key trong JSON (vd: 'CNN_MODEL_PATH')
        - force: nếu True thì tải lại dù file đã tồn tại
        """
        if model_key not in self.cfg.paths:
            logging.error(f"❌ Model key '{model_key}' not found in config.")
            return

        info = self.cfg.paths[model_key]
        full_path = info["full_path"]

        # Nếu file tồn tại và không ép tải lại
        if os.path.exists(full_path) and not force:
            logging.info(f"✅ {model_key} already exists at {full_path}")
            return

        # Xử lý tải model
        logging.info(f"⬇️ Starting manual download for {model_key}")
        self._check_and_download_single(model_key, info)
