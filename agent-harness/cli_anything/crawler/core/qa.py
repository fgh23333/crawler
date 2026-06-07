"""QA module - generate and manage QA pairs from concept graphs."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .crawl import find_project_root


def generate_qa(graph_file: Optional[str] = None, config_file: Optional[str] = None) -> dict:
    """Generate QA pairs from a concept graph using MemCube Political.

    Args:
        graph_file: Path to concept graph JSON file
        config_file: Path to config.yaml

    Returns:
        dict with status, QA counts
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    memcube_dir = root / "memcube-political"
    if not memcube_dir.exists():
        return {"status": "error", "message": f"MemCube directory not found: {memcube_dir}"}

    cfg = config_file or str(memcube_dir / "config" / "config.yaml")

    env = {**__import__("os").environ, "PYTHONPATH": str(memcube_dir / "src")}
    args = [sys.executable, str(memcube_dir / "src" / "main.py"), "--stage", "qa-generation", "--config", cfg]
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=7200, cwd=str(memcube_dir), env=env,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        }

    return {
        "status": "done",
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
    }


def run_full_pipeline(config_file: Optional[str] = None) -> dict:
    """Run the full MemCube pipeline (concept expansion + QA generation).

    Args:
        config_file: Path to config.yaml

    Returns:
        dict with status, results
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    memcube_dir = root / "memcube-political"
    if not memcube_dir.exists():
        return {"status": "error", "message": f"MemCube directory not found: {memcube_dir}"}

    cfg = config_file or str(memcube_dir / "config" / "config.yaml")

    env = {**__import__("os").environ, "PYTHONPATH": str(memcube_dir / "src")}
    args = [sys.executable, str(memcube_dir / "src" / "main.py"), "--stage", "all", "--config", cfg]
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=14400, cwd=str(memcube_dir), env=env,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "returncode": result.returncode,
            "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        }

    return {
        "status": "done",
        "stdout": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
    }


def qa_stats(qa_file: Optional[str] = None) -> dict:
    """Show statistics for QA data.

    Args:
        qa_file: Path to QA JSON file

    Returns:
        dict with status, QA statistics
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    if qa_file:
        f = Path(qa_file)
    else:
        # Search for QA files
        search_dir = root / "memcube-political" / "data"
        candidates = list(search_dir.rglob("*qa*.json"))
        if not candidates:
            candidates = list((root / "memos").glob("*qa*.json"))
        if not candidates:
            return {"status": "error", "message": "No QA files found"}
        f = max(candidates, key=lambda p: p.stat().st_mtime)

    if not f.exists():
        return {"status": "error", "message": f"QA file not found: {f}"}

    data = json.loads(f.read_text(encoding="utf-8"))

    stats = {
        "file": str(f),
        "file_size_kb": round(f.stat().st_size / 1024, 1),
        "total_count": len(data) if isinstance(data, list) else 1,
    }

    if isinstance(data, list) and data:
        # Analyze by subject
        by_subject = {}
        by_type = {}
        for item in data:
            subj = item.get("subject_name", "unknown")
            qtype = item.get("question_type", "unknown")
            by_subject[subj] = by_subject.get(subj, 0) + 1
            by_type[qtype] = by_type.get(qtype, 0) + 1
        stats["by_subject"] = by_subject
        stats["by_type"] = by_type

    return {"status": "done", **stats}
