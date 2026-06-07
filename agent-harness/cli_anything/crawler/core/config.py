"""Config module - manage configuration and check environment."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .crawl import find_project_root


def show_config() -> dict:
    """Show current configuration.

    Returns:
        dict with status, config info
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    memcube_dir = root / "memcube-political"
    config_file = memcube_dir / "config" / "config.yaml"
    api_file = memcube_dir / "config" / "api_keys.yaml"

    info = {
        "project_root": str(root),
        "memcube_dir": str(memcube_dir),
        "memcube_exists": memcube_dir.exists(),
        "config_exists": config_file.exists(),
        "api_config_exists": api_file.exists(),
        "node_modules_exists": (root / "node_modules").exists(),
    }

    # Check Python dependencies
    try:
        import yaml
        info["yaml"] = True
    except ImportError:
        info["yaml"] = False

    try:
        import openai
        info["openai"] = True
    except ImportError:
        info["openai"] = False

    try:
        import networkx
        info["networkx"] = True
    except ImportError:
        info["networkx"] = False

    try:
        import neo4j
        info["neo4j"] = True
    except ImportError:
        info["neo4j"] = False

    try:
        import qdrant_client
        info["qdrant_client"] = True
    except ImportError:
        info["qdrant_client"] = False

    # Check Node.js dependencies
    try:
        result = subprocess.run(["node", "-e", "require('axios')"], capture_output=True, timeout=10)
        info["node_axios"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["node_axios"] = False

    try:
        result = subprocess.run(["node", "-e", "require('cheerio')"], capture_output=True, timeout=10)
        info["node_cheerio"] = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["node_cheerio"] = False

    # Read config if available
    if config_file.exists():
        try:
            import yaml
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            info["config"] = {k: str(v)[:100] for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}
        except Exception as e:
            info["config_error"] = str(e)

    return {"status": "done", **info}


def check_env() -> dict:
    """Check the environment for required dependencies.

    Returns:
        dict with status, checks
    """
    checks = []

    # Python version
    checks.append({
        "name": "Python",
        "installed": True,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "required": "3.8+",
        "ok": sys.version_info >= (3, 8),
    })

    # Node.js
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10)
        node_ver = result.stdout.strip()
        checks.append({
            "name": "Node.js",
            "installed": True,
            "version": node_ver,
            "required": "14+",
            "ok": True,
        })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        checks.append({"name": "Node.js", "installed": False, "ok": False})

    # Python packages
    packages = [
        ("openai", "OpenAI SDK"),
        ("yaml", "PyYAML"),
        ("networkx", "NetworkX"),
        ("neo4j", "Neo4j Driver"),
        ("qdrant_client", "Qdrant Client"),
        ("loguru", "Loguru"),
        ("tqdm", "tqdm"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    for pkg, label in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            checks.append({"name": label, "installed": True, "version": ver, "ok": True})
        except ImportError:
            checks.append({"name": label, "installed": False, "ok": False})

    # Node.js packages
    for pkg, label in [("axios", "axios"), ("cheerio", "cheerio"), ("jsdom", "jsdom")]:
        try:
            result = subprocess.run(
                ["node", "-e", f"console.log(require('{pkg}').version || 'installed')"],
                capture_output=True, text=True, timeout=10,
            )
            ver = result.stdout.strip()
            checks.append({"name": f"Node: {label}", "installed": True, "version": ver, "ok": True})
        except (FileNotFoundError, subprocess.TimeoutExpired):
            checks.append({"name": f"Node: {label}", "installed": False, "ok": False})

    ok_count = sum(1 for c in checks if c["ok"])
    return {
        "status": "done",
        "total": len(checks),
        "passed": ok_count,
        "failed": len(checks) - ok_count,
        "checks": checks,
    }


def test_api() -> dict:
    """Test API configuration.

    Returns:
        dict with status, test results
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    memcube_dir = root / "memcube-political"
    test_script = memcube_dir / "scripts" / "test_api_simple.py"

    if not test_script.exists():
        return {"status": "error", "message": f"Test script not found: {test_script}"}

    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True, text=True, timeout=60, cwd=str(memcube_dir),
    )

    return {
        "status": "done" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
    }
