import threading
from typing import Dict, Any
from datetime import datetime


class InMemoryProgressTracker:
    """Thread-safe in-memory progress tracking with automatic cleanup"""

    def __init__(self, cleanup_interval_minutes: int = 60):
        self.progress_data: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.cleanup_interval = cleanup_interval_minutes
        self.last_cleanup = datetime.now()

    def _cleanup_expired_entries(self):
        """Remove entries older than cleanup_interval"""
        with self.lock:
            now = datetime.now()
            if (now - self.last_cleanup).total_seconds() < self.cleanup_interval * 60:
                return

            expired_keys = []
            for doc_id, data in self.progress_data.items():
                updated_at = datetime.fromisoformat(
                    data.get("updated_at", now.isoformat())
                )
                if (now - updated_at).total_seconds() > self.cleanup_interval * 60:
                    expired_keys.append(doc_id)

            for key in expired_keys:
                del self.progress_data[key]

            self.last_cleanup = now

    def update_progress(
        self,
        document_id: str,
        status: str,
        progress: int,
        current_stage: str,
        error_message: str = None,
        processing_results: Dict = None,
    ):
        """Update document processing progress"""
        with self.lock:
            self._cleanup_expired_entries()

            self.progress_data[document_id] = {
                "status": status,
                "progress_percentage": progress,
                "current_stage": current_stage,
                "error_message": error_message,
                "processing_results": processing_results or {},
                "updated_at": datetime.now().isoformat(),
            }

    def get_progress(self, document_id: str) -> Dict[str, Any]:
        """Get document processing progress"""
        with self.lock:
            self._cleanup_expired_entries()

            if document_id not in self.progress_data:
                return {"error": "Document not found"}

            return self.progress_data[document_id]

    def delete_progress(self, document_id: str) -> bool:
        """Delete progress data for a document"""
        with self.lock:
            if document_id in self.progress_data:
                del self.progress_data[document_id]
                return True
            return False
