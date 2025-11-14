from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse
import hashlib
import logging
import warnings

from model_inferences import (
    XGBoostInference, RFInference,
    WordCNNInference, WordCNNLSTMInference, CharCNNInference, CharCNNLSTMInference, CNNHybridInference,
    ALBERTInference, MobileBERTInference, TinyLlamaInference
)
from configs import FirebaseConfigs
from firebase_client import FirebaseClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore")

INFERENCE_MAP = {
    "xgb": XGBoostInference,
    "rf": RFInference,
    "wordcnn": WordCNNInference,
    "wordcnn_lstm": WordCNNLSTMInference,
    "charcnn": CharCNNInference,
    "charcnn_lstm": CharCNNLSTMInference,
    "cnn_hybrid": CNNHybridInference,
    "albert": ALBERTInference,
    "mobile_bert": MobileBERTInference,
    "tiny_llama": TinyLlamaInference,
}

app = Flask(__name__)
CORS(app)
firebase_cfg = FirebaseConfigs()
firebase = FirebaseClient(firebase_cfg.info["cred_path"],
                          firebase_cfg.info["project_id"],
                          firebase_cfg.info["collections"])


def domain_hash(url: str) -> str:
    domain = urlparse(url).netloc.lower() or url
    return hashlib.sha256(domain.encode()).hexdigest()


def run_model(model_name: str, url: str, threshold: float):
    model_cls = INFERENCE_MAP.get(model_name)
    if not model_cls:
        raise ValueError(f"Unknown model '{model_name}'")
    model = model_cls(url)
    pred, proba = model.predict(threshold)
    return {"url": url, "pred": int(pred), "proba": float(proba), "model": model_name}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url = data.get("url")
    model_name = data.get("model", "xgb").lower()
    threshold = float(data.get("threshold", 0.5))
    if not url:
        return jsonify({"error": "Missing 'url'"}), 400

    url = url.strip()
    d_hash = domain_hash(url)

    # --- Kiểm tra cache ---
    cached = firebase.get(d_hash, model_name)
    if cached:
        cached["cached"] = True
        return jsonify(cached), 200

    # --- Dự đoán mới ---
    try:
        result = run_model(model_name, url, threshold)
        firebase.save(d_hash, model_name, result)
        result["cached"] = False
        return jsonify(result), 200
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/healthz")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
