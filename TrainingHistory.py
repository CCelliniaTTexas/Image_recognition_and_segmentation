import json
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path


class TrainingHistoryStore:
    """File-backed semantic segmentation training history."""

    def __init__(self, history_dir=None):
        self.history_dir = Path(history_dir) if history_dir else Path.home() / ".adl_tool" / "training_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create_run(self, metadata):
        started_at = metadata.get("started_at") or datetime.now().isoformat(timespec="seconds")
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "run_id": run_id,
            "status": "running",
            "started_at": started_at,
            "ended_at": None,
            "parameters": metadata.get("parameters", {}),
            "runtime": metadata.get("runtime", {}),
            "results": {},
            "artifacts": {},
            "error": None,
        }
        self._write_json(run_dir / "metadata.json", record)
        self._write_json(run_dir / "artifacts.json", {})
        (run_dir / "metrics.jsonl").touch(exist_ok=True)
        return run_id

    def update_metadata(self, run_id, updates):
        with self._lock:
            metadata = self.get_run(run_id) or {}
            self._deep_update(metadata, self._make_json_safe(updates))
            self._write_json(self._run_dir(run_id) / "metadata.json", metadata)

    def append_metric(self, run_id, metric):
        if not run_id:
            return
        metric = self._make_json_safe(metric)
        with self._lock:
            with open(self._run_dir(run_id) / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(metric, ensure_ascii=False) + "\n")

    def update_artifacts(self, run_id, artifacts):
        if not run_id:
            return
        artifacts = self._make_json_safe(artifacts)
        with self._lock:
            run_dir = self._run_dir(run_id)
            existing = self._read_json(run_dir / "artifacts.json", default={})
            self._deep_update(existing, artifacts)
            self._write_json(run_dir / "artifacts.json", existing)
            metadata = self._read_json(run_dir / "metadata.json", default={})
            metadata.setdefault("artifacts", {})
            self._deep_update(metadata["artifacts"], artifacts)
            self._write_json(run_dir / "metadata.json", metadata)

    def finish_run(self, run_id, status, results=None, error=None):
        if not run_id:
            return
        updates = {
            "status": status,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "results": results or {},
            "error": error,
        }
        self.update_metadata(run_id, updates)

    def list_runs(self):
        runs = []
        for run_dir in self.history_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata = self._read_json(run_dir / "metadata.json", default=None)
            if metadata:
                runs.append(metadata)
        runs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
        return runs

    def get_run(self, run_id):
        return self._read_json(self._run_dir(run_id) / "metadata.json", default=None)

    def load_metrics(self, run_id):
        metrics_path = self._run_dir(run_id) / "metrics.jsonl"
        if not metrics_path.exists():
            return []

        metrics = []
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return metrics

    def clear(self):
        if self.history_dir.exists():
            shutil.rmtree(self.history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id):
        return self.history_dir / run_id

    @staticmethod
    def _read_json(path, default):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    @classmethod
    def _write_json(cls, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cls._make_json_safe(data), f, ensure_ascii=False, indent=2)

    @classmethod
    def _make_json_safe(cls, value):
        if isinstance(value, dict):
            return {str(k): cls._make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._make_json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        if isinstance(value, Path):
            return str(value)
        return value

    @classmethod
    def _deep_update(cls, target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_update(target[key], value)
            else:
                target[key] = value
