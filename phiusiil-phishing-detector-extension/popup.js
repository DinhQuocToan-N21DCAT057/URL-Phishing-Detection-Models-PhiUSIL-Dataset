const API_BASE = "http://127.0.0.1:8000/predict";

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  // Cache element
  els.currentUrl = document.getElementById("current-url");
  els.modelSelect = document.getElementById("model-select");
  els.thresholdInput = document.getElementById("threshold-input");
  els.scanBtn = document.getElementById("scan-btn");
  els.status = document.getElementById("status");
  els.result = document.getElementById("result");
  els.error = document.getElementById("error");

  els.resModel = document.getElementById("res-model");
  els.resPred = document.getElementById("res-pred");
  els.resProba = document.getElementById("res-proba");
  els.resThreshold = document.getElementById("res-threshold");
  els.resCached = document.getElementById("res-cached");
  els.resCreated = document.getElementById("res-created");
  els.resUrl = document.getElementById("res-url");

  // Load config đã lưu (model, threshold)
  chrome.storage.sync.get(["phiusiil_model", "phiusiil_threshold"], (data) => {
    if (data.phiusiil_model) {
      els.modelSelect.value = data.phiusiil_model;
    }
    if (typeof data.phiusiil_threshold === "number") {
      els.thresholdInput.value = data.phiusiil_threshold.toFixed(2);
    }
  });

  // Lấy URL tab hiện tại
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const tabUrl = tab && tab.url ? tab.url : "";
  els.currentUrl.textContent = tabUrl || "(Không lấy được URL)";
  els.resUrl.textContent = tabUrl;

  els.scanBtn.addEventListener("click", () => runScan(tabUrl));
});

function setStatus(type, message) {
  els.status.classList.remove("hidden", "safe", "phishing", "loading");
  els.status.textContent = message;

  if (type === "safe") els.status.classList.add("safe");
  else if (type === "phishing") els.status.classList.add("phishing");
  else if (type === "loading") els.status.classList.add("loading");
}

function clearMessages() {
  els.error.classList.add("hidden");
  els.error.textContent = "";
  els.result.classList.add("hidden");
}

async function runScan(url) {
  clearMessages();

  if (!url) {
    els.error.textContent = "Không lấy được URL tab hiện tại.";
    els.error.classList.remove("hidden");
    return;
  }

  const model = els.modelSelect.value;
  let threshold = parseFloat(els.thresholdInput.value);
  if (Number.isNaN(threshold)) threshold = 0.5;
  threshold = Math.min(Math.max(threshold, 0), 1);

  // Lưu cấu hình
  chrome.storage.sync.set({
    phiusiil_model: model,
    phiusiil_threshold: threshold
  });

  els.thresholdInput.value = threshold.toFixed(2);

  setStatus("loading", "Đang gửi tới API...");
  els.scanBtn.disabled = true;

  try {
    const payload = {
      url: url,
      model: model,
      threshold: threshold
    };

    const res = await fetch(API_BASE, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API trả về HTTP ${res.status}: ${text}`);
    }

    const data = await res.json();
    showResult(data, threshold);
  } catch (err) {
    els.error.textContent = "Lỗi gọi API: " + err.message;
    els.error.classList.remove("hidden");
    setStatus("phishing", "Không đánh giá được – kiểm tra API.");
  } finally {
    els.scanBtn.disabled = false;
  }
}

function showResult(data, threshold) {
  const pred = Number(data.pred) || 0;      // 0 = benign, 1 = phishing
  const proba = Number(data.proba) || 0;
  const model = data.model || "?";
  const cached = !!data.cached;
  const createdAt = data.created_at || "";
  const url = data.url || "";

  const isPhishing = pred === 1;

  if (isPhishing) {
    setStatus(
      "phishing",
      ` Phishing / URL đáng ngờ (p = ${proba.toFixed(3)})`
    );
  } else {
    setStatus(
      "safe",
      ` Có vẻ an toàn (p = ${proba.toFixed(3)})`
    );
  }

  els.resModel.textContent = model;
  els.resPred.textContent = isPhishing ? "1 (phishing)" : "0 (benign)";
  els.resProba.textContent = proba.toFixed(4);
  els.resThreshold.textContent = threshold.toFixed(2);
  els.resCached.textContent = cached ? "Yes (dùng cache Firestore)" : "No (dự đoán mới)";

  if (createdAt) {
    try {
      const dt = new Date(createdAt);
      els.resCreated.textContent = dt.toLocaleString();
    } catch {
      els.resCreated.textContent = createdAt;
    }
  } else {
    els.resCreated.textContent = "(không có)";
  }

  els.resUrl.textContent = url;

  els.result.classList.remove("hidden");
}
