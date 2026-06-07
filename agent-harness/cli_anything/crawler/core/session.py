"""Session module - manage persistent sessions and state."""

import json
from pathlib import Path
from typing import Optional, Any


DEFAULT_SESSION_DIR = Path.home() / ".crawler_sessions"


class Session:
    """Manage a persistent session for crawler operations."""

    def __init__(self, name: str = "default", session_dir: Optional[str] = None):
        self.name = name
        self.session_dir = Path(session_dir) if session_dir else DEFAULT_SESSION_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.file = self.session_dir / f"{name}.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"name": self.name, "created": None, "variables": {}, "history": []}

    def _save(self):
        self.file.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get("variables", {}).get(key, default)

    def set(self, key: str, value: Any):
        if "variables" not in self._data:
            self._data["variables"] = {}
        self._data["variables"][key] = value
        self._save()

    def add_history(self, command: str, result: str = ""):
        if self._data.get("created") is None:
            from datetime import datetime
            self._data["created"] = datetime.now().isoformat()
        if "history" not in self._data:
            self._data["history"] = []
        self._data["history"].append({"command": command, "result": result[:500]})
        # Keep last 100 entries
        self._data["history"] = self._data["history"][-100:]
        self._save()

    def list_sessions() -> list:
        """List all available sessions."""
        DEFAULT_SESSION_DIR.mkdir(parents=True, exist_ok=True)
        sessions = []
        for f in sorted(DEFAULT_SESSION_DIR.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "name": f.stem,
                    "created": data.get("created"),
                    "commands": len(data.get("history", [])),
                })
            except (json.JSONDecodeError, OSError):
                sessions.append({"name": f.stem, "error": "unreadable"})
        return sessions

    def delete(self):
        if self.file.exists():
            self.file.unlink()

    def info(self) -> dict:
        return {
            "name": self.name,
            "file": str(self.file),
            "created": self._data.get("created"),
            "variables": dict(self._data.get("variables", {})),
            "history_count": len(self._data.get("history", [])),
        }
