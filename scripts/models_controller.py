import os
import threading
import logging
import warnings
import time
from functools import wraps
import config as cfg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

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

class W2Vec:
    _instance = None
    _lock = threading.Lock()
    _model = None
    _dim = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(W2Vec, cls).__new__(cls)
        return cls._instance

    def __init__(self, url, dim) -> None:
        pass
    
    @classmethod
    def _load_model(cls):
        """Load XGB Model."""
        model_path = cfg.Paths.WORD2VEC_MODEL_PATH
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
        
        model = Word2Vec.load(cfg.Paths.WORD2VEC_MODEL_PATH)
        logging.info(f"Model loaded successfully from {model_path}")
        return model

    @classmethod
    def get_model(cls, reload=False):
        return cls._model

from xgboost import XGBClassifier

class XGBoost:
    _instance = None
    _lock = threading.Lock()
    _model = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(XGBoost, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        pass

    @classmethod
    def _load_model(cls):
        """Load XGB Model."""
        model_path = cfg.Paths.XGB_MODEL_PATH
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
        
        model = XGBClassifier()
        model.load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    
    @classmethod
    def get_model(cls, reload=False):
        """
        Lấy instance XGB lưu ở cache.
        Nếu `reload=True`, model sẽ được load lại từ file.
        """
        with cls._lock:
            if cls._model is None or reload:
                cls._model = cls._load_model()
            return cls._model
    
    def predict(self, X):
        model = self.get_model()
        return model.predict(X)
