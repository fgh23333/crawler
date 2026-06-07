"""Crawl module - login and fetch political theory questions from the exam system."""

import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional


SUBJECTS = {
    "XiIntro": {"subjectId": "1752935841845477392", "name": "习近平新时代中国特色社会主义思想概论"},
    "Marx": {"subjectId": "1748167277460586496", "name": "马克思主义基本原理"},
    "MaoIntro": {"subjectId": "1748168736914800651", "name": "毛泽东思想和中国特色社会主义理论体系概论"},
    "Political": {"subjectId": "1781216923707506688", "name": "思想道德与法治"},
    "CMH": {"subjectId": "1748168736914800640", "name": "中国近现代史纲要"},
    "NCH": {"subjectId": "1776854236110258176", "name": "新中国史"},
    "SDH": {"subjectId": "1752935841845477376", "name": "社会主义发展史"},
    "ORH": {"subjectId": "1752935841845477384", "name": "改革开放史"},
    "CCPH": {"subjectId": "1798740810791911424", "name": "中共党史"},
}

DEFAULT_BRANCH_ID = "1705139277953761280"
BASE_URL = "http://222.73.57.153:6571"

# Credential storage file
CRED_FILE = Path.home() / ".crawler_credentials.json"


def find_project_root() -> Optional[Path]:
    """Find the crawler project root by looking for known markers."""
    current = Path.cwd()
    for _ in range(20):
        if (current / "package.json").exists() or (current / "memcube-political").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def save_credentials(student_num: str, password: str):
    """Save credentials to local file."""
    CRED_FILE.write_text(json.dumps({
        "student_num": student_num,
        "password": password,
        "branch_id": DEFAULT_BRANCH_ID,
    }), encoding="utf-8")


def load_credentials() -> Optional[dict]:
    """Load saved credentials."""
    if CRED_FILE.exists():
        try:
            return json.loads(CRED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def login(student_num: str, password: str, branch_id: Optional[str] = None) -> dict:
    """Login to the exam system.

    Args:
        student_num: Student number
        password: Password
        branch_id: Branch ID (default: 1705139277953761280)

    Returns:
        dict with status, student_id, name, etc.
    """
    bid = branch_id or DEFAULT_BRANCH_ID
    login_data = json.dumps({
        "branchId": bid,
        "studentNum": student_num,
        "practiceType": "1",
        "password": password,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/login/practiceLogin",
            data=login_data,
            headers={"Content-Type": "application/json;charset=UTF-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if result.get("code") != "200":
            return {"status": "error", "message": result.get("msg", "Login failed")}

        student_info = result["data"]["studentInfo"]
        student_id = student_info["id"]

        # Save credentials for later use
        save_credentials(student_num, password)

        return {
            "status": "done",
            "student_id": student_id,
            "name": student_info.get("name", ""),
            "student_number": student_info.get("studentNumber", ""),
            "branch_name": student_info.get("branchName", ""),
            "message": "Login successful",
        }
    except Exception as e:
        return {"status": "error", "message": f"Login request failed: {e}"}


def fetch_questions(subject: str, output_dir: Optional[str] = None,
                    iterations: int = 1000, student_id: Optional[str] = None,
                    student_num: Optional[str] = None, password: Optional[str] = None) -> dict:
    """Fetch questions for a specific subject using the exam API.

    Args:
        subject: Subject code (e.g., 'Marx', 'XiIntro')
        output_dir: Output directory for JSON file
        iterations: Number of fetch iterations (default 1000)
        student_id: Student ID (if not provided, will login first)
        student_num: Student number (for login if student_id not provided)
        password: Password (for login if student_id not provided)

    Returns:
        dict with status, subject, output_path, question_count
    """
    if subject not in SUBJECTS:
        return {"status": "error", "message": f"Unknown subject: {subject}. Valid: {list(SUBJECTS.keys())}"}

    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    out_dir = Path(output_dir) if output_dir else root / "new"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject}.json"

    # Get student_id - login if needed
    sid = student_id
    if not sid:
        creds = load_credentials()
        if creds:
            login_result = login(creds["student_num"], creds["password"], creds.get("branch_id"))
            if login_result["status"] == "done":
                sid = login_result["student_id"]
            else:
                return {"status": "error", "message": f"Auto-login failed: {login_result['message']}. Provide --student-id or login first."}
        elif student_num and password:
            login_result = login(student_num, password)
            if login_result["status"] == "done":
                sid = login_result["student_id"]
            else:
                return {"status": "error", "message": f"Login failed: {login_result['message']}"}
        else:
            return {"status": "error", "message": "No student_id provided and no saved credentials. Run 'crawl login' first or provide --student-num and --password."}

    # Build and run the fetcher script with checkpoint every 100 iterations
    params = {
        "branchId": DEFAULT_BRANCH_ID,
        "chapterId": "",
        "studentId": sid,
        "subjectId": SUBJECTS[subject]["subjectId"],
    }
    script = f'''
import json, sys, os, urllib.request

params = {json.dumps(params)}
base_url = "{BASE_URL}"
iterations = {iterations}
output_path = r"{str(out_path)}"
checkpoint_every = 100

# Load existing data for resume
seen = set()
existing = []
if os.path.exists(output_path):
    try:
        existing = json.load(open(output_path, "r", encoding="utf-8"))
        for item in existing:
            seen.add(item.get("id", ""))
        print(f"Resumed from existing file: {{len(existing)}} questions", file=sys.stderr)
    except:
        existing = []

results = list(existing)
errors = 0
start = 0

for i in range(iterations):
    try:
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(
            base_url + "/examinationInfo/getPracticeInfo",
            data=data,
            headers={{"Content-Type": "application/json;charset=utf-8"}},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
            if raw.get("code") != "200":
                print(f"API error at iteration {{i+1}}: {{raw.get('msg', 'unknown')}}", file=sys.stderr)
                errors += 1
                if errors >= 10:
                    print("Too many errors, stopping.", file=sys.stderr)
                    break
                continue
            paper_content = json.loads(raw["data"]["paperStore"]["paperContent"])
            batch = []
            for key in ["panduan", "danxuan", "duoxuan", "tiankong"]:
                for item in paper_content.get(key, {{}}).get("children", []):
                    if item.get("id", "") not in seen:
                        seen.add(item["id"])
                        batch.append(item)
            results.extend(batch)
            errors = 0
        # Checkpoint every N iterations
        if (i + 1) % checkpoint_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            print(f"[{{'{subject}'}}] Checkpoint at {{i+1}}/{{iterations}}: {{len(results)}} unique questions", file=sys.stderr)
    except Exception as e:
        print(f"Error at iteration {{i+1}}: {{e}}", file=sys.stderr)
        errors += 1
        if errors >= 10:
            print("Too many errors, stopping.", file=sys.stderr)
            break

# Final save
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print(json.dumps({{"status": "done", "subject": "{subject}", "output_path": str(output_path), "question_count": len(results), "iterations": i+1, "new_questions": len(results) - len(existing)}}))
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=max(7200, iterations * 5)
    )

    if result.returncode != 0 and not result.stdout.strip():
        return {"status": "error", "message": result.stderr.strip()}

    # Parse the last JSON line from stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"status": "done", "subject": subject, "output_path": str(out_path), "message": result.stdout.strip()}


def fetch_all(output_dir: Optional[str] = None, iterations: int = 1000,
              student_num: Optional[str] = None, password: Optional[str] = None) -> dict:
    """Fetch questions for all subjects.

    Returns:
        dict with status, results list, total_questions
    """
    results = []
    total = 0
    for subject in SUBJECTS:
        r = fetch_questions(subject, output_dir, iterations,
                            student_num=student_num, password=password)
        results.append(r)
        if r.get("question_count"):
            total += r["question_count"]

    return {
        "status": "done",
        "results": results,
        "total_questions": total,
        "subjects_fetched": len(results),
    }


def crawl_status(data_dir: Optional[str] = None) -> dict:
    """Show crawl status for all subjects.

    Returns:
        dict with status, subjects info (file exists, question count, file size)
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    search_dir = Path(data_dir) if data_dir else root / "new"

    subjects_info = {}
    for code, info in SUBJECTS.items():
        json_file = search_dir / f"{code}.json"
        entry = {
            "code": code,
            "name": info["name"],
            "file": str(json_file),
            "exists": json_file.exists(),
        }
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                entry["question_count"] = len(data)
                entry["file_size_kb"] = round(json_file.stat().st_size / 1024, 1)
            except (json.JSONDecodeError, OSError):
                entry["question_count"] = 0
                entry["file_size_kb"] = 0
        subjects_info[code] = entry

    total_questions = sum(s.get("question_count", 0) for s in subjects_info.values())
    files_exist = sum(1 for s in subjects_info.values() if s["exists"])

    return {
        "status": "done",
        "subjects": subjects_info,
        "total_questions": total_questions,
        "files_exist": files_exist,
        "total_subjects": len(SUBJECTS),
    }
