import os

class Paths:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(CURRENT_DIR)

    XGB_MODEL_PATH = os.path.join(ROOT_DIR, "models", "")
    RF_MODEL_PATH = os.path.join(ROOT_DIR, "models", "")
    CNN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "")
    CNN_LSTM_PATH = os.path.join(ROOT_DIR, "models", "")
    CHAR_CNN_PATH = os.path.join(ROOT_DIR, "models", "")
    CHAR_CNN_LSTM_PATH = os.path.join(ROOT_DIR, "models", "")
    CNN_HYBRID_PATH = os.path.join(ROOT_DIR, "models", "")
    ALBERT_PATH = os.path.join(ROOT_DIR, "models", "")
    MOBILE_BERT_PATH = os.path.join(ROOT_DIR, "models", "")
    