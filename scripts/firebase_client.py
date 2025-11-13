# firebase_client.py
import os
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseClient:
    """Đóng gói Firestore cho lưu & truy vấn kết quả dự đoán."""

    def __init__(self, creds_path: str, project_id: str, collection: str = "predictions", ttl_days: int = 30):
        if not firebase_admin._apps:
            cred = credentials.Certificate(creds_path)
            firebase_admin.initialize_app(cred, {"projectId": project_id})
        self.db = firestore.client()
        self.col = self.db.collection(collection)
        self.ttl = timedelta(days=ttl_days)

    def _key(self, domain_hash: str, model: str) -> str:
        return f"{domain_hash}:{model}"

    def get(self, domain_hash: str, model: str):
        doc = self.col.document(self._key(domain_hash, model)).get()
        if not doc.exists:
            return None
        data = doc.to_dict()
        created = datetime.fromisoformat(data.get("created_at"))
        if datetime.utcnow() - created > self.ttl:
            self.col.document(self._key(domain_hash, model)).delete()
            return None
        return data

    def save(self, domain_hash: str, model: str, payload: dict):
        payload["created_at"] = datetime.utcnow().isoformat()
        self.col.document(self._key(domain_hash, model)).set(payload)
