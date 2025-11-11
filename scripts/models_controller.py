import os
import threading
import logging
import warnings
import time
import numpy as np
import hashlib
import json

from functools import wraps
from model_downloader import ModelDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

loader = ModelDownloader()


def timer(func):
    """Record execution time of any functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        if args and hasattr(args[0], "exec_time"):
            args[0].exec_time += elapsed_time
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds, cumulative exec_time: {args[0].exec_time:.2f} seconds"
            )
        else:
            logging.info(
                f"Function '{func.__name__}' took {elapsed_time:.2f} seconds (no instance with exec_time found)"
            )
        return result  # Return the original function's result

    return wrapper


from gensim.models import Word2Vec
from urllib.parse import urlparse
import re


class W2Vec:
    """PhiUSIIL Word2Vec"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(W2Vec, cls).__new__(cls)
        return cls._instance

    def __init__(self, dim=50):
        self._dim = dim
        self._tokens = []

    def _load_model(self):
        """Load Word2Vec model từ file config."""
        loader.download_single("WORD2VEC_MODEL_PATH")
        model_path = loader.cfg.paths["WORD2VEC_MODEL_PATH"]["full_path"]

        model = Word2Vec.load(model_path)
        logging.info(f"✅ Word2Vec model loaded successfully from {model_path}")
        return model

    def get_model(self, reload=False):
        """Lấy model Word2Vec (cache)."""
        with self._lock:
            if self._model is None or reload:
                self._model = self._load_model()
            return self._model

    def _preprocess(self, url: str):
        """Chuyển URL thành danh sách token hợp lệ."""
        if not url or not isinstance(url, (str, bytes)):
            raise ValueError("URL phải là chuỗi hợp lệ")

        self._tokens.clear()
        self._url = url.decode('utf-8') if isinstance(url, bytes) else url.lower()
        parsed = urlparse(self._url)

        # Tách domain, path, query
        domain_parts = parsed.netloc.split('.') if parsed.netloc else []
        path_parts = [p for p in parsed.path.split('/') if p]
        query_parts = parsed.query.split('&') if parsed.query else []

        tokens = domain_parts + path_parts + query_parts
        # Lọc ký tự đặc biệt
        clean_tokens = [re.sub(r'[^a-z0-9\-\/.=]', '', t) for t in tokens if t]
        self._tokens.extend(clean_tokens)

        return self._tokens

    def get_vector(self, url):
        """Trích vector trung bình từ các token trong URL."""
        model = self.get_model()
        tokens = self._preprocess(url)

        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            vec = np.mean(vectors, axis=0)
        else:
            vec = np.zeros(self._dim)

        return vec


from xgboost import XGBClassifier

from xgboost import XGBClassifier
import numpy as np
import logging
import threading


class XGBoost:
    """PhiUSIIL XGBoost"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(XGBoost, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.url = None

    @classmethod
    def _load_model(cls):
        """Load XGBoost model từ file config."""
        loader.download_single("XGB_MODEL_PATH")
        model_path = loader.cfg.paths["XGB_MODEL_PATH"]["full_path"]

        model = XGBClassifier()
        model.load_model(model_path)
        logging.info(f"✅ XGBoost model loaded successfully from {model_path}")
        return model

    @classmethod
    def get_model(cls, reload=False):
        """Lấy instance XGB lưu ở cache."""
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model

    def predict(self, url, threshold=0.5):
        """Dự đoán nhị phân với xác suất."""
        model = self.get_model()
        self.url = url
        w2v = W2Vec()
        vector = w2v.get_vector(url).reshape(1, -1)
        proba = model.predict_proba(vector)[0][1]
        pred = int(proba >= threshold)
        return pred, proba

    def predict_json(self, url, threshold=0.5):
        """Trả kết quả dự đoán theo format JSON chuẩn."""
        pred, proba = self.predict(url, threshold)
        sha256_hash = hashlib.sha256(self.url.encode()).hexdigest()

        result = {
            sha256_hash: {
                "url": self.url,
                "pred": pred,
                "proba": round(float(proba), 4),
                "threshold": threshold
            }
        }

        return json.dumps(result, indent=4)


class RF:
    """PhiUSIIL Random Forest"""
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(RF, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.url = None

