"""Unit tests for crawler CLI core modules."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project structure mimicking the crawler project."""
    # Create directories
    (tmp_path / "new").mkdir()
    (tmp_path / "2025-05-27" / "solved").mkdir(parents=True)
    (tmp_path / "2025-05-27" / "rewrite").mkdir()
    (tmp_path / "merge").mkdir()
    (tmp_path / "memos").mkdir()
    (tmp_path / "memcube-political" / "config").mkdir(parents=True)
    (tmp_path / "memcube-political" / "data").mkdir(parents=True)
    (tmp_path / "memcube-political" / "src").mkdir(parents=True)
    (tmp_path / "memcube-political" / "scripts").mkdir(parents=True)
    (tmp_path / "package.json").write_text('{"name": "crawler"}')

    # Create sample solved data
    solved_data = [
        {
            "id": "001",
            "title": "马克思主义的本质是什么",
            "standardAnswer": "A",
            "options": "A.科学理论|B.宗教|C.哲学|D.艺术",
            "score": "2",
        },
        {
            "id": "002",
            "title": "中国特色社会主义是科学社会主义",
            "standardAnswer": "正确",
            "options": "",
            "score": "2",
        },
        {
            "id": "003",
            "title": "以下哪些属于马克思主义基本原理",
            "standardAnswer": "ABCD",
            "options": "A.唯物史观|B.剩余价值论|C.阶级斗争|D.科学社会主义",
            "score": "2",
        },
        {
            "id": "004",
            "title": "___是马克思主义中国化的第一次历史性飞跃",
            "standardAnswer": "毛泽东思想",
            "options": "",
            "score": "2",
        },
    ]
    (tmp_path / "2025-05-27" / "solved" / "MarxSolved.json").write_text(
        json.dumps(solved_data, ensure_ascii=False), encoding="utf-8"
    )

    # Create sample rewrite data
    rewrite_sc = [
        {
            "questionStem": "马克思主义的本质是什么",
            "option": ["科学理论", "宗教", "哲学", "艺术"],
            "answer": "A",
            "id": "001",
            "likeFlag": False,
            "markFlag": False,
            "abbreviationSubject": "Marx",
        }
    ]
    rewrite_rw = [
        {
            "questionStem": "中国特色社会主义是科学社会主义",
            "option": ["正确", "错误"],
            "answer": "正确",
            "id": "002",
            "likeFlag": False,
            "markFlag": False,
            "abbreviationSubject": "Marx",
        }
    ]
    (tmp_path / "2025-05-27" / "rewrite" / "Marx_singleChoice.json").write_text(
        json.dumps(rewrite_sc, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "2025-05-27" / "rewrite" / "Marx_rightWrong.json").write_text(
        json.dumps(rewrite_rw, ensure_ascii=False), encoding="utf-8"
    )

    # Create sample new/rewrite for merge
    (tmp_path / "new" / "rewrite").mkdir()
    (tmp_path / "new" / "rewrite" / "Marx_singleChoice.json").write_text(
        json.dumps(rewrite_sc, ensure_ascii=False), encoding="utf-8"
    )

    # Create sample transformed data
    transformed = [
        {
            "id": "Text-0",
            "question": "马克思主义的本质是什么\nAnswer Choices: (A) 科学理论 (B) 宗教 (C) 哲学 (D) 艺术",
            "options": [{"letter": "A", "content": "科学理论"}],
            "label": ["A"],
            "subject_name": "马克思主义基本原理",
            "question_type": "singleChoice",
        }
    ]
    (tmp_path / "memos" / "transformed_political_data.json").write_text(
        json.dumps(transformed, ensure_ascii=False), encoding="utf-8"
    )

    # Create concept graph
    graph = {
        "nodes": [
            {"id": "c1", "name": "马克思主义"},
            {"id": "c2", "name": "唯物主义"},
            {"id": "c3", "name": "辩证法"},
        ],
        "edges": [
            {"source": "c1", "target": "c2", "relation": "包含"},
            {"source": "c1", "target": "c3", "relation": "包含"},
        ],
    }
    (tmp_path / "memcube-political" / "data" / "final_concept_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False), encoding="utf-8"
    )

    # Create config
    config = {
        "logging": {"level": "INFO", "format": "{message}", "rotation": "1 day", "retention": "7 days"},
        "paths": {
            "logs_dir": str(tmp_path / "memcube-political" / "logs"),
            "seed_concepts": str(tmp_path / "memos" / "political_seed_concepts.json"),
            "results_dir": str(tmp_path / "memcube-political" / "data"),
            "concept_graph_dir": str(tmp_path / "memcube-political" / "data"),
        },
        "concept_expansion": {"batch_size": 10, "max_workers": 4},
    }
    (tmp_path / "memcube-political" / "config" / "config.yaml").write_text(
        "# config\n", encoding="utf-8"
    )

    return tmp_path


# ── Test find_project_root ─────────────────────────────────────────────────

class TestFindProjectRoot:
    def test_finds_root_with_package_json(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_path):
            from cli_anything.crawler.core.crawl import find_project_root
            result = find_project_root()
            assert result == tmp_path

    def test_returns_none_when_not_found(self, tmp_path):
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_path):
            from cli_anything.crawler.core.crawl import find_project_root
            result = find_project_root()
            assert result is None


# ── Test crawl module ──────────────────────────────────────────────────────

class TestCrawlStatus:
    def test_status_with_data(self, tmp_project):
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.crawl import crawl_status
            result = crawl_status()
            assert result["status"] == "done"
            assert result["total_subjects"] == 9

    def test_status_no_project_root(self, tmp_path):
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_path):
            from cli_anything.crawler.core.crawl import crawl_status
            result = crawl_status()
            assert result["status"] == "error"


class TestFetchQuestions:
    def test_invalid_subject(self, tmp_project):
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.crawl import fetch_questions
            result = fetch_questions("InvalidSubject")
            assert result["status"] == "error"
            assert "Unknown subject" in result["message"]

    def test_valid_subject_structure(self, tmp_project):
        with patch("cli_anything.crawler.core.crawl.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.crawl import fetch_questions
            # Mock subprocess to avoid actual network call
            with patch("cli_anything.crawler.core.crawl.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout='{"status": "done", "subject": "Marx", "question_count": 42, "output_path": "/tmp/test.json"}',
                    stderr="",
                )
                result = fetch_questions("Marx", iterations=1)
                assert result["status"] == "done"
                assert result["question_count"] == 42


# ── Test data module ───────────────────────────────────────────────────────

class TestRewriteData:
    def test_rewrite_creates_classified_files(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import rewrite_data
            out_dir = tmp_project / "test_rewrite_out"
            result = rewrite_data(
                input_dir=str(tmp_project / "2025-05-27" / "solved"),
                output_dir=str(out_dir),
            )
            assert result["status"] == "done"
            assert result["total_questions"] > 0
            assert (out_dir / "Marx_singleChoice.json").exists()

    def test_rewrite_nonexistent_input(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import rewrite_data
            result = rewrite_data(input_dir=str(tmp_project / "nonexistent"))
            assert result["status"] == "error"


class TestClassifyQuestions:
    def test_classify(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            # Create test data for classification
            classify_dir = tmp_project / "classify_test"
            classify_dir.mkdir()
            test_data = [
                {"option": ["A", "B"], "answer": "A", "id": "1"},
                {"option": [], "answer": "fill", "id": "2"},
                {"option": ["A", "B", "C", "D"], "answer": "A", "id": "3"},
                {"option": ["A", "B", "C", "D"], "answer": "AB", "id": "4"},
            ]
            (classify_dir / "test.json").write_text(json.dumps(test_data), encoding="utf-8")

            from cli_anything.crawler.core.data import classify_questions
            result = classify_questions(input_dir=str(classify_dir))
            assert result["status"] == "done"
            stats = result["classifications"]["test.json"]
            assert stats["rightWrong"] == 1
            assert stats["fillingBlank"] == 1
            assert stats["singleChoice"] == 1
            assert stats["multipleChoice"] == 1


class TestMergeData:
    def test_merge(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import merge_data
            result = merge_data(
                old_dir=str(tmp_project / "new" / "rewrite"),
                new_dir=str(tmp_project / "2025-05-27" / "rewrite"),
                output_dir=str(tmp_project / "test_merge"),
            )
            assert result["status"] == "done"
            assert "Marx_singleChoice.json" in result["files"]

    def test_merge_missing_dir(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import merge_data
            result = merge_data(old_dir=str(tmp_project / "nonexistent"))
            assert result["status"] == "error"


class TestTransformData:
    def test_transform(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import transform_data
            out_file = tmp_project / "test_transformed.json"
            result = transform_data(
                input_dir=str(tmp_project / "2025-05-27" / "rewrite"),
                output_file=str(out_file),
            )
            assert result["status"] == "done"
            assert result["total_questions"] >= 2
            assert out_file.exists()


class TestDataStats:
    def test_stats(self, tmp_project):
        with patch("cli_anything.crawler.core.data.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.data import data_stats
            result = data_stats(data_dir=str(tmp_project / "2025-05-27" / "rewrite"))
            assert result["status"] == "done"
            assert result["total_files"] >= 2


# ── Test concept module ────────────────────────────────────────────────────

class TestExtractConcepts:
    def test_extract(self, tmp_project):
        with patch("cli_anything.crawler.core.concept.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.concept import extract_concepts
            out_dir = tmp_project / "test_concepts"
            result = extract_concepts(
                input_file=str(tmp_project / "memos" / "transformed_political_data.json"),
                output_dir=str(out_dir),
            )
            assert result["status"] == "done"
            assert result["total_concepts"] > 0
            assert (out_dir / "political_seed_concepts.json").exists()
            assert (out_dir / "political_seed_concepts.txt").exists()

    def test_extract_missing_file(self, tmp_project):
        with patch("cli_anything.crawler.core.concept.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.concept import extract_concepts
            result = extract_concepts(input_file=str(tmp_project / "nonexistent.json"))
            assert result["status"] == "error"


class TestGraphInfo:
    def test_graph_info(self, tmp_project):
        with patch("cli_anything.crawler.core.concept.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.concept import graph_info
            result = graph_info(
                graph_file=str(tmp_project / "memcube-political" / "data" / "final_concept_graph.json")
            )
            assert result["status"] == "done"
            assert result["node_count"] == 3
            assert result["edge_count"] == 2

    def test_graph_info_no_file(self, tmp_project):
        with patch("cli_anything.crawler.core.concept.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.concept import graph_info
            result = graph_info(graph_file=str(tmp_project / "nonexistent.json"))
            assert result["status"] == "error"


# ── Test QA module ─────────────────────────────────────────────────────────

class TestQAStats:
    def test_qa_stats(self, tmp_project):
        # Create a QA file
        qa_data = [
            {"subject_name": "马克思主义基本原理", "question_type": "singleChoice"},
            {"subject_name": "马克思主义基本原理", "question_type": "multipleChoice"},
        ]
        qa_file = tmp_project / "test_qa.json"
        qa_file.write_text(json.dumps(qa_data, ensure_ascii=False), encoding="utf-8")

        with patch("cli_anything.crawler.core.qa.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.qa import qa_stats
            result = qa_stats(qa_file=str(qa_file))
            assert result["status"] == "done"
            assert result["total_count"] == 2


# ── Test config module ─────────────────────────────────────────────────────

class TestShowConfig:
    def test_show_config(self, tmp_project):
        with patch("cli_anything.crawler.core.config.Path.cwd", return_value=tmp_project):
            from cli_anything.crawler.core.config import show_config
            result = show_config()
            assert result["status"] == "done"
            assert result["project_root"] == str(tmp_project)
            assert result["memcube_exists"] is True


class TestCheckEnv:
    def test_check_env(self):
        from cli_anything.crawler.core.config import check_env
        result = check_env()
        assert result["status"] == "done"
        assert result["total"] > 0
        assert result["passed"] > 0


# ── Test session module ────────────────────────────────────────────────────

class TestSession:
    def test_session_set_get(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        s = Session(name="test", session_dir=str(tmp_path / "sessions"))
        s.set("key1", "value1")
        assert s.get("key1") == "value1"

    def test_session_persistence(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        s1 = Session(name="persist", session_dir=str(tmp_path / "sessions"))
        s1.set("foo", "bar")
        s2 = Session(name="persist", session_dir=str(tmp_path / "sessions"))
        assert s2.get("foo") == "bar"

    def test_session_history(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        s = Session(name="hist", session_dir=str(tmp_path / "sessions"))
        s.add_history("crawl fetch --subject Marx", "status=done")
        info = s.info()
        assert info["history_count"] == 1

    def test_session_list(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        Session.list_sessions()  # Should not error
        s = Session(name="list_test", session_dir=str(tmp_path / "sessions"))
        s.set("x", "y")
        sessions = Session.list_sessions()
        # Note: list_sessions uses DEFAULT_SESSION_DIR, not tmp_path

    def test_session_delete(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        s = Session(name="del_test", session_dir=str(tmp_path / "sessions"))
        s.set("x", "y")
        assert s.file.exists()
        s.delete()
        assert not s.file.exists()

    def test_session_info(self, tmp_path):
        from cli_anything.crawler.core.session import Session
        s = Session(name="info_test", session_dir=str(tmp_path / "sessions"))
        info = s.info()
        assert info["name"] == "info_test"


# ── Test output formatting ─────────────────────────────────────────────────

class TestOutputFormatting:
    def test_json_mode(self):
        from cli_anything.crawler.utils.output import format_output
        data = {"status": "done", "count": 42}
        result = format_output(data, json_mode=True)
        parsed = json.loads(result)
        assert parsed["count"] == 42

    def test_human_mode(self):
        from cli_anything.crawler.utils.output import format_output
        data = {"status": "done", "count": 42, "name": "test"}
        result = format_output(data, json_mode=False)
        assert "42" in result
        assert "test" in result

    def test_error_mode(self):
        from cli_anything.crawler.utils.output import format_output
        data = {"status": "error", "message": "something went wrong"}
        result = format_output(data, json_mode=False)
        assert "Error:" in result
        assert "something went wrong" in result

    def test_json_error_mode(self):
        from cli_anything.crawler.utils.output import format_output
        data = {"status": "error", "message": "fail"}
        result = format_output(data, json_mode=True)
        parsed = json.loads(result)
        assert parsed["status"] == "error"
