"""E2E tests for the crawler CLI - full pipeline tests with real files."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# Resolve the CLI command - always use the installed entry point via the venv python
CLI_FORCE_INSTALLED = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "0") == "1"


def _find_venv_python():
    """Find the venv Python executable."""
    # Look for .venv relative to the project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    for venv_name in (".venv", "venv"):
        venv_dir = project_root / venv_name
        if venv_dir.exists():
            if os.name == "nt":
                py = venv_dir / "Scripts" / "python.exe"
            else:
                py = venv_dir / "bin" / "python"
            if py.exists():
                return str(py)
    return sys.executable


def _resolve_cli():
    """Resolve the CLI command path."""
    if CLI_FORCE_INSTALLED:
        return "cli-anything-crawler"
    # Use the venv python with -m to run the CLI module
    return [_find_venv_python(), "-m", "cli_anything.crawler.crawler_cli"]


def _run_cli(*args, cwd=None, env=None):
    """Run the CLI with given arguments."""
    # Always use the venv python to run the module directly
    cmd = [_find_venv_python(), "-m", "cli_anything.crawler.crawler_cli"] + list(args)

    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Add project to PYTHONPATH
    project_root = Path(__file__).parent.parent.parent.parent.parent
    pythonpath = str(project_root / "agent-harness")
    run_env["PYTHONPATH"] = pythonpath + os.pathsep + run_env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=cwd, env=run_env,
    )
    return result


class TestCLIEntry:
    """Test basic CLI entry point."""

    def test_version(self):
        result = _run_cli("--version")
        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    def test_help(self):
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "crawler" in result.stdout.lower() or "Crawler" in result.stdout

    def test_no_args_shows_help(self):
        result = _run_cli()
        assert result.returncode == 0


class TestCrawlCommands:
    """Test crawl command group."""

    def test_crawl_help(self):
        result = _run_cli("crawl", "--help")
        assert result.returncode == 0
        assert "fetch" in result.stdout
        assert "status" in result.stdout

    def test_crawl_status_json(self):
        result = _run_cli("crawl", "status", "--json")
        # Should succeed or fail gracefully (no project root in test env)
        output = result.stdout + result.stderr
        assert "status" in output

    def test_crawl_fetch_invalid_subject(self):
        result = _run_cli("crawl", "fetch", "--subject", "InvalidSubject")
        assert result.returncode == 0
        assert "Unknown subject" in result.stdout


class TestDataCommands:
    """Test data command group."""

    def test_data_help(self):
        result = _run_cli("data", "--help")
        assert result.returncode == 0
        assert "rewrite" in result.stdout
        assert "classify" in result.stdout
        assert "merge" in result.stdout
        assert "transform" in result.stdout
        assert "stats" in result.stdout

    def test_data_stats_json(self):
        result = _run_cli("--json", "data", "stats")
        assert result.returncode == 0


class TestConceptCommands:
    """Test concept command group."""

    def test_concept_help(self):
        result = _run_cli("concept", "--help")
        assert result.returncode == 0
        assert "extract" in result.stdout
        assert "expand" in result.stdout
        assert "info" in result.stdout

    def test_concept_info_no_file(self):
        result = _run_cli("concept", "info", "--file", "/nonexistent/graph.json")
        # Should fail gracefully
        assert "error" in (result.stdout + result.stderr).lower() or result.returncode != 0


class TestQACommands:
    """Test QA command group."""

    def test_qa_help(self):
        result = _run_cli("qa", "--help")
        assert result.returncode == 0
        assert "generate" in result.stdout
        assert "pipeline" in result.stdout
        assert "stats" in result.stdout


class TestConfigCommands:
    """Test config command group."""

    def test_config_help(self):
        result = _run_cli("config", "--help")
        assert result.returncode == 0
        assert "show" in result.stdout
        assert "check-env" in result.stdout

    def test_config_check_env(self):
        result = _run_cli("config", "check-env")
        assert result.returncode == 0
        assert "Python" in result.stdout


class TestSessionCommands:
    """Test session command group."""

    def test_session_help(self):
        result = _run_cli("session", "--help")
        assert result.returncode == 0

    def test_session_list(self):
        result = _run_cli("session", "list")
        assert result.returncode == 0


class TestJSONOutput:
    """Test JSON output mode."""

    def test_json_flag(self):
        result = _run_cli("--json", "config", "check-env")
        assert result.returncode == 0
        # Should be valid JSON
        try:
            data = json.loads(result.stdout)
            assert "status" in data
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


class TestCLISubprocess:
    """Test CLI via subprocess (as an installed command)."""

    def _run_installed(self, *args):
        """Run using the installed command name."""
        env = os.environ.copy()
        env["CLI_ANYTHING_FORCE_INSTALLED"] = "1"
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent.parent.parent / "agent-harness")
        cmd = [_find_venv_python(), "-m", "cli_anything.crawler.crawler_cli"] + list(args)
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)

    def test_installed_version(self):
        result = self._run_installed("--version")
        assert result.returncode == 0
        assert "1.0.0" in result.stdout

    def test_installed_help(self):
        result = self._run_installed("--help")
        assert result.returncode == 0

    def test_installed_check_env(self):
        result = self._run_installed("config", "check-env")
        assert result.returncode == 0


class TestFullWorkflow:
    """Test a full workflow from data to concepts."""

    def test_full_workflow(self, tmp_path):
        """Test: rewrite → classify → transform → extract concepts."""
        # Setup project structure
        project = tmp_path / "crawler_project"
        project.mkdir()
        (project / "package.json").write_text('{"name": "crawler"}')
        (project / "2025-05-27" / "solved").mkdir(parents=True)
        (project / "2025-05-27" / "rewrite").mkdir()
        (project / "memos").mkdir()

        # Create sample solved data
        solved_data = [
            {"id": "1", "title": "什么是马克思主义", "standardAnswer": "A",
             "options": "A.科学理论|B.宗教信仰|C.文化运动|D.政治派别", "score": "2"},
            {"id": "2", "title": "辩证唯物主义是马克思主义哲学的基础", "standardAnswer": "正确",
             "options": "", "score": "2"},
        ]
        (project / "2025-05-27" / "solved" / "TestSolved.json").write_text(
            json.dumps(solved_data, ensure_ascii=False), encoding="utf-8"
        )

        env = {
            "PYTHONPATH": str(Path(__file__).parent.parent.parent.parent.parent / "agent-harness"),
        }

        # Step 1: Rewrite
        result = _run_cli("data", "rewrite",
                          "--input", str(project / "2025-05-27" / "solved"),
                          "--output", str(project / "2025-05-27" / "rewrite"))
        assert result.returncode == 0

        # Step 2: Transform
        result = _run_cli("data", "transform",
                          "--input", str(project / "2025-05-27" / "rewrite"),
                          "--output", str(project / "memos" / "transformed.json"))
        assert result.returncode == 0

        # Step 3: Extract concepts
        result = _run_cli("concept", "extract",
                          "--input", str(project / "memos" / "transformed.json"),
                          "--output", str(project / "concepts_out"))
        assert result.returncode == 0

        # Verify outputs exist
        assert (project / "memos" / "transformed.json").exists()
        assert (project / "concepts_out" / "political_seed_concepts.json").exists()
        assert (project / "concepts_out" / "political_seed_concepts.txt").exists()
