# PhiUSIL URL Phishing Detection System

## 1. Project Summary

This project implements a comprehensive URL phishing detection system capable of classifying URLs into categories such as benign and phishing. The dataset used is [PhiUSILL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) and additional URL from other sources, in total more than 400,000 URLs. It leverages a variety of machine learning and deep learning models to ensure high accuracy and robustness.

Key features include:
- **Diverse Model Architecture**: Utilizes traditional ML models (XGBoost, Random Forest), Deep Learning models (CNN, LSTM varieties), and modern Transformers (ALBERT, MobileBERT, TinyLlama).
- **Real-time Inference**: Designed for fast prediction using a Flask-based API.
- **Data Handling**: Includes pipelines for feature extraction and processing large-scale datasets.
- **Scalability**: Capable of handling significant volumes of URL requests.

## 2. Model Performance Comparison

The following table summarizes the quantitative evaluation metrics for the models used in this project.

| Model Name | F1-Score | Precision | Recall | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | 0.88 | 0.89 | 0.88 | 0.90 | 0.92 |
| **Random Forest** | 0.86 | 0.87 | 0.86 | 0.84 | 0.89 |
| **WordCNN** | 0.89 | 0.93 | 0.93 | 0.96 | 0.90 |
| **WordCNN LSTM** | 0.89 | 0.92 | 0.95 | 0.89 | 0.92 |
| **CharCNN** | 0.95 | 0.94 | 0.96 | 0.98 | 0.99 |
| **CharCNN LSTM** | 0.96 | 0.95 | 0.97 | 0.98 | 0.99 |
| **CNN Hybrid** | 0.93 | 0.89 | 0.97 | 0.97 | 0.97 |
| **ALBERT** | 0.96 | 0.96 | 0.97 | 0.99 | 0.99 |
| **MobileBERT** | 0.96 | 0.95 | 0.98 | 0.99 | 0.99 |
| **TinyLlama LoRA** | N/A | N/A | N/A | N/A | N/A |

## 3. Installation & Configuration

### Prerequisites
- **Python**: Version 3.8 or higher is recommended.
- **Node.js**: Required if using the web crawler or extension components.
- **CUDA**: Recommended if running Deep Learning models (Transformers/CNNs) for faster inference.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DinhQuocToan-N21DCAT057/URL-Phishing-Detection-Models-PhiUSIL-Dataset.git
   cd URL-Phishing-Detection-Models-PhiUSIL-Dataset
   ```

2. **Install Python Dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   # Create virtual environment (optional)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Install Node.js Dependencies (Optional):**
   If you plan to work with the crawler or extension:
   ```bash
   npm install
   ```

4. **Docker (Optional):**
   If you plan to use Docker for containerization:
   ```bash
   docker build -t url-phishing-detection .
   docker run -p 8000:8000 url-phishing-detection
   ```

   or use our image from Docker Hub:
   ```bash
   docker pull dinhtoan2157/phiusiil-url-detector:latest
   docker run -p 8000:8000 dinhtoan2157/phiusiil-url-detector:latest
   ```

### Configuration

- **Firebase**: The application uses Firebase for caching/logging. Ensure you have your `serviceAccountKey.json` or equivalent credentials configured in the `secrets/` directory or as specified in `scripts/configs.py` and your Firestore database is running with your specific `collections` (configured in `path_map.json`).
- **Model Files**: The system may attempt to download models automatically via `scripts/model_downloader.py`. Ensure you have internet access or place the pre-trained models in the correctly mapped directories (referenced in `path_map.json`).

### Usage

To start the API server for inference:

```bash
python scripts/app.py
```

The server will start on `http://0.0.0.0:8000`.

**API Endpoint:** `POST /predict`
- **Body:** `{"url": "http://example.com", "model": "xgb"}`