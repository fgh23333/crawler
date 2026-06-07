"""Data processing module - rewrite, classify, merge, transform, and analyze question data."""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from .crawl import find_project_root, SUBJECTS


SUBJECT_MAPPING = {
    "XiIntro": "习近平新时代中国特色社会主义思想概论",
    "Marx": "马克思主义基本原理",
    "MaoIntro": "毛泽东思想和中国特色社会主义理论体系概论",
    "Political": "思想道德与法治",
    "CMH": "中国近现代史纲要",
    "NCH": "新中国史",
    "SDH": "社会主义发展史",
    "ORH": "改革开放史",
    "CCPH": "中共党史",
}

QUESTION_TYPES = ["singleChoice", "multipleChoice", "rightWrong", "fillingBlank", "subject"]


def rewrite_data(input_dir: Optional[str] = None, output_dir: Optional[str] = None) -> dict:
    """Rewrite raw solved data into classified format (singleChoice, multipleChoice, etc.).

    Uses the same logic as rewritePython.py.

    Args:
        input_dir: Directory containing solved JSON files (e.g., 2025-05-27/solved/)
        output_dir: Output directory for classified files

    Returns:
        dict with status, per-subject counts
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    in_dir = Path(input_dir) if input_dir else root / "2025-05-27" / "solved"
    out_dir = Path(output_dir) if output_dir else root / "2025-05-27" / "rewrite"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        return {"status": "error", "message": f"Input directory not found: {in_dir}"}

    stats = {}
    total = 0

    for filepath in sorted(in_dir.glob("*.json")):
        subject_name = filepath.stem.replace("Solved", "").replace("solved", "")
        sc, mc, rw, fb, subj = [], [], [], [], []
        result_all = []

        try:
            # Try streaming parse with ijson, fall back to json
            try:
                import ijson
                with open(filepath, "r", encoding="utf-8") as f:
                    for item in ijson.items(f, "item"):
                        nn = item.get("standardAnswer", "").replace(" ", "")
                        nnt = item.get("title", "").replace(" ", "")
                        nnid = item.get("id", "").replace(" ", "")

                        temp = {
                            "questionStem": nnt,
                            "option": [],
                            "answer": nn,
                            "id": nnid,
                            "likeFlag": False,
                            "markFlag": False,
                            "abbreviationSubject": subject_name,
                        }

                        if nn in ("正确", "错误"):
                            temp["option"] = ["正确", "错误"]
                            rw.append(temp)
                        elif nn and 65 <= ord(nn[0]) <= 90:
                            opts = item.get("options", "").replace(" ", "").replace(".", "").split("|")
                            temp["option"] = opts
                            if len(nn) == 1 and len(opts) == 4:
                                sc.append(temp)
                            else:
                                mc.append(temp)
                        else:
                            temp["option"] = ""
                            temp["answer"] = nn.replace("|", "，")
                            fb.append(temp)
                        result_all.append(temp)
            except ImportError:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                for item in data:
                    nn = item.get("standardAnswer", "").replace(" ", "")
                    nnt = item.get("title", "").replace(" ", "")
                    nnid = item.get("id", "").replace(" ", "")

                    temp = {
                        "questionStem": nnt,
                        "option": [],
                        "answer": nn,
                        "id": nnid,
                        "likeFlag": False,
                        "markFlag": False,
                        "abbreviationSubject": subject_name,
                    }

                    if nn in ("正确", "错误"):
                        temp["option"] = ["正确", "错误"]
                        rw.append(temp)
                    elif nn and 65 <= ord(nn[0]) <= 90:
                        opts = item.get("options", "").replace(" ", "").replace(".", "").split("|")
                        temp["option"] = opts
                        if len(nn) == 1 and len(opts) == 4:
                            sc.append(temp)
                        else:
                            mc.append(temp)
                    else:
                        temp["option"] = ""
                        temp["answer"] = nn.replace("|", "，")
                        fb.append(temp)
                    result_all.append(temp)

            # Write classified files
            def _write(qtype, data_list):
                if data_list:
                    p = out_dir / f"{subject_name}_{qtype}.json"
                    p.write_text(json.dumps(data_list, ensure_ascii=False, indent=2), encoding="utf-8")

            _write("singleChoice", sc)
            _write("multipleChoice", mc)
            _write("rightWrong", rw)
            _write("fillingBlank", fb)
            _write("subject", result_all)

            stats[subject_name] = {
                "singleChoice": len(sc),
                "multipleChoice": len(mc),
                "rightWrong": len(rw),
                "fillingBlank": len(fb),
                "subject": len(result_all),
            }
            total += len(result_all)

        except Exception as e:
            stats[subject_name] = {"error": str(e)}

    return {"status": "done", "stats": stats, "total_questions": total, "output_dir": str(out_dir)}


def classify_questions(input_dir: Optional[str] = None) -> dict:
    """Classify questions by type based on option count and answer length.

    Uses the same logic as classify.js.

    Args:
        input_dir: Directory containing JSON files to classify

    Returns:
        dict with status, files processed, classifications
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    in_dir = Path(input_dir) if input_dir else root / "merge"
    if not in_dir.exists():
        return {"status": "error", "message": f"Input directory not found: {in_dir}"}

    stats = {}
    for filepath in sorted(in_dir.glob("*.json")):
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            classifications = {"singleChoice": 0, "multipleChoice": 0, "rightWrong": 0, "fillingBlank": 0}

            for obj in data:
                opts = obj.get("option", [])
                answer = obj.get("answer", "")

                if len(opts) == 2:
                    obj["type"] = "rightWrong"
                    classifications["rightWrong"] += 1
                elif len(opts) == 0:
                    obj["type"] = "fillingBlank"
                    classifications["fillingBlank"] += 1
                elif len(str(answer)) == 1:
                    obj["type"] = "singleChoice"
                    classifications["singleChoice"] += 1
                else:
                    obj["type"] = "multipleChoice"
                    classifications["multipleChoice"] += 1

            filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            stats[filepath.name] = classifications
        except Exception as e:
            stats[filepath.name] = {"error": str(e)}

    return {"status": "done", "files_processed": len(stats), "classifications": stats}


def merge_data(old_dir: Optional[str] = None, new_dir: Optional[str] = None,
               output_dir: Optional[str] = None) -> dict:
    """Merge old and new question data, deduplicating by content.

    Uses the same logic as merge.js.

    Args:
        old_dir: Old data directory
        new_dir: New data directory
        output_dir: Output directory for merged files

    Returns:
        dict with status, merge results per file
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    old_d = Path(old_dir) if old_dir else root / "new" / "rewrite"
    new_d = Path(new_dir) if new_dir else root / "2025-05-27" / "rewrite"
    out_d = Path(output_dir) if output_dir else root / "merge"
    out_d.mkdir(parents=True, exist_ok=True)

    if not old_d.exists():
        return {"status": "error", "message": f"Old directory not found: {old_d}"}
    if not new_d.exists():
        return {"status": "error", "message": f"New directory not found: {new_d}"}

    results = {}
    for old_file in sorted(old_d.glob("*.json")):
        new_file = new_d / old_file.name
        out_file = out_d / old_file.name

        if not new_file.exists():
            results[old_file.name] = {"status": "skipped", "reason": "not in new directory"}
            continue

        old_data = json.loads(old_file.read_text(encoding="utf-8")) if old_file.exists() else []
        new_data = json.loads(new_file.read_text(encoding="utf-8"))

        # Keep old items that still exist in new data
        updated_old = [item for item in old_data
                       if any(json.dumps(item, sort_keys=True) == json.dumps(n, sort_keys=True) for n in new_data)]
        # Add new items not in old
        new_items = [item for item in new_data
                     if not any(json.dumps(item, sort_keys=True) == json.dumps(o, sort_keys=True) for o in updated_old)]
        merged = updated_old + new_items

        out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        results[old_file.name] = {
            "old_count": len(old_data),
            "new_count": len(new_data),
            "merged_count": len(merged),
            "added": len(new_items),
        }

    return {"status": "done", "output_dir": str(out_d), "files": results}


def transform_data(input_dir: Optional[str] = None, output_file: Optional[str] = None) -> dict:
    """Transform classified data into standard QA format for concept extraction.

    Uses the same logic as data_transformer.js.

    Args:
        input_dir: Directory containing classified JSON files (rewrite/)
        output_file: Output JSON file path

    Returns:
        dict with status, total questions, per-subject counts
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    in_dir = Path(input_dir) if input_dir else root / "2025-05-27" / "rewrite"
    out_file = Path(output_file) if output_file else root / "memos" / "transformed_political_data.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        return {"status": "error", "message": f"Input directory not found: {in_dir}"}

    results = []
    index = 0
    subject_stats = {}

    for subject in SUBJECTS:
        for qtype in QUESTION_TYPES:
            filepath = in_dir / f"{subject}_{qtype}.json"
            if not filepath.exists():
                continue

            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                subject_name = SUBJECT_MAPPING.get(subject, subject)

                for question in data:
                    if not question.get("questionStem", "").strip():
                        continue

                    question_text = question["questionStem"]
                    options = []
                    if question.get("option") and isinstance(question["option"], list):
                        for i, opt in enumerate(question["option"]):
                            if opt and str(opt).strip():
                                options.append({
                                    "letter": chr(65 + i),
                                    "content": str(opt).strip(),
                                })

                    label = str(question.get("answer", "A"))
                    if label == "正确":
                        label = "A"
                    elif label == "错误":
                        label = "B"

                    if not label or not all(c in "ABCDEFGHIJ" for c in label.upper()):
                        label = "A"

                    full_question = question_text
                    if options:
                        opts_text = " ".join(f"({o['letter']}) {o['content']}" for o in options)
                        full_question = f"{question_text}\nAnswer Choices: {opts_text}"

                    results.append({
                        "id": f"Text-{index}",
                        "question": full_question,
                        "options": options,
                        "label": [label.upper()],
                        "subject_name": subject_name,
                        "question_type": qtype,
                    })
                    index += 1

                subject_stats[f"{subject}_{qtype}"] = len(data)
            except Exception as e:
                subject_stats[f"{subject}_{qtype}"] = {"error": str(e)}

    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Per-subject summary
    subj_summary = {}
    for r in results:
        sn = r["subject_name"]
        subj_summary[sn] = subj_summary.get(sn, 0) + 1

    return {
        "status": "done",
        "total_questions": len(results),
        "output_file": str(out_file),
        "per_subject": subj_summary,
    }


def data_stats(data_dir: Optional[str] = None) -> dict:
    """Show statistics for question data in a directory.

    Args:
        data_dir: Directory to scan for JSON files

    Returns:
        dict with status, per-file stats, totals
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    search_dir = Path(data_dir) if data_dir else root
    file_stats = []
    total_questions = 0

    for json_file in sorted(search_dir.rglob("*.json")):
        # Skip large data files (>50MB) for speed
        if json_file.stat().st_size > 50 * 1024 * 1024:
            file_stats.append({
                "file": str(json_file.relative_to(search_dir)),
                "size_kb": round(json_file.stat().st_size / 1024, 1),
                "questions": "skipped (large file)",
            })
            continue

        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            count = len(data) if isinstance(data, list) else 1
            total_questions += count
            file_stats.append({
                "file": str(json_file.relative_to(search_dir)),
                "size_kb": round(json_file.stat().st_size / 1024, 1),
                "questions": count,
            })
        except (json.JSONDecodeError, OSError):
            file_stats.append({
                "file": str(json_file.relative_to(search_dir)),
                "size_kb": round(json_file.stat().st_size / 1024, 1),
                "questions": 0,
                "error": "invalid json",
            })

    return {
        "status": "done",
        "directory": str(search_dir),
        "total_files": len(file_stats),
        "total_questions": total_questions,
        "files": file_stats,
    }
