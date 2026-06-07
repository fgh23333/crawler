#!/usr/bin/env python3
"""Parallel fetcher with auto-relogin and checkpointing."""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "http://222.73.57.153:6571"
BRANCH_ID = "1705139277953761280"

SUBJECTS = {
    "XiIntro": "1752935841845477392",
    "Marx": "1748167277460586496",
    "MaoIntro": "1748168736914800651",
    "Political": "1781216923707506688",
    "CMH": "1748168736914800640",
    "NCH": "1776854236110258176",
    "SDH": "1752935841845477376",
    "ORH": "1752935841845477384",
    "CCPH": "1798740810791911424",
}

STUDENT_NUM = sys.argv[1] if len(sys.argv) > 1 else "2352613"
PASSWORD = sys.argv[2] if len(sys.argv) > 2 else "2352613"
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
OUTPUT_DIR = sys.argv[4] if len(sys.argv) > 4 else "new"
CHECKPOINT = 100


def do_login():
    """Login and return student_id."""
    data = json.dumps({
        "branchId": BRANCH_ID, "studentNum": STUDENT_NUM,
        "practiceType": "1", "password": PASSWORD
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/login/practiceLogin", data=data,
        headers={"Content-Type": "application/json;charset=UTF-8"}, method="POST")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == "200":
                sid = result["data"]["studentInfo"]["id"]
                print(f"[LOGIN] Success, studentId={sid}", flush=True)
                return sid
            else:
                print(f"[LOGIN] Failed: {result.get('msg')}, retry {attempt+1}/5", flush=True)
        except Exception as e:
            print(f"[LOGIN] Error: {e}, retry {attempt+1}/5", flush=True)
        time.sleep(3)
    return None


def _do_request(params, subject, i):
    """Make a single API request, return list of questions or None."""
    try:
        import requests
        resp = requests.post(
            f"{BASE_URL}/examinationInfo/getPracticeInfo",
            json=params, timeout=30)
        raw = resp.json()
        if raw.get("code") != "200":
            return None, raw.get("msg", "unknown")
        paper = json.loads(raw["data"]["paperStore"]["paperContent"])
        items = []
        for key in ["panduan", "danxuan", "duoxuan", "tiankong"]:
            items.extend(paper.get(key, {}).get("children", []))
        return items, None
    except Exception as e:
        return None, str(e)


def fetch_one(subject, subject_id, student_id):
    """Fetch questions for one subject using 4 concurrent workers with batching."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    out_path = Path(OUTPUT_DIR) / f"{subject}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing
    seen = set()
    results = []
    if out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
            for item in results:
                seen.add(item.get("id", ""))
        except:
            results = []

    params = {
        "branchId": BRANCH_ID, "chapterId": "",
        "studentId": student_id, "subjectId": subject_id
    }

    lock = threading.Lock()
    errors = [0]
    new_count = [0]
    sid = [student_id]
    done = [0]
    stale = [0]  # consecutive iterations with 0 new questions
    STALE_LIMIT = 500  # stop after this many iterations with no new questions

    def worker(i):
        items, err = _do_request(dict(params), subject, i)
        with lock:
            done[0] += 1
            if items is None:
                if err and ("登录" in err or "过期" in err or "401" in err):
                    new_sid = do_login()
                    if new_sid:
                        sid[0] = new_sid
                        params["studentId"] = new_sid
                errors[0] += 1
            else:
                batch_new = 0
                for item in items:
                    iid = item.get("id", "")
                    if iid and iid not in seen:
                        seen.add(iid)
                        results.append(item)
                        batch_new += 1
                if batch_new > 0:
                    new_count[0] += batch_new
                    stale[0] = 0
                else:
                    stale[0] += 1
                errors[0] = 0

            # Checkpoint
            if done[0] % CHECKPOINT == 0:
                out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
                print(f"[{subject}] {done[0]}/{ITERATIONS} | total={len(results)} | new={new_count[0]} | stale={stale[0]}", flush=True)

            # Early termination
            if stale[0] >= STALE_LIMIT:
                print(f"[{subject}] No new questions for {STALE_LIMIT} iterations, stopping early.", flush=True)
                # Signal other workers to stop
                return

    # Process in batches of 200 to avoid memory issues
    BATCH = 200
    early_stop = [False]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch_start in range(0, ITERATIONS, BATCH):
            if early_stop[0]:
                print(f"[{subject}] Early stop triggered.", flush=True)
                break
            batch_end = min(batch_start + BATCH, ITERATIONS)
            futures = [pool.submit(worker, i) for i in range(batch_start, batch_end)]
            for f in futures:
                result = f.result()
                if result is None:  # early stop signal
                    early_stop[0] = True
                    break

    # Final save
    out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"[{subject}] DONE. total={len(results)}, new={new_count[0]}", flush=True)
    return subject, len(results), new_count[0]


def main():
    print(f"=== Parallel Fetcher ===", flush=True)
    print(f"Subjects: {len(SUBJECTS)}, Iterations: {ITERATIONS}, Output: {OUTPUT_DIR}", flush=True)

    # Login once first
    sid = do_login()
    if not sid:
        print("Initial login failed, exiting.")
        sys.exit(1)

    # Fetch subjects sequentially (each with 4 internal threads)
    print(f"Starting fetch for {len(SUBJECTS)} subjects (4 threads each)...", flush=True)
    start_time = time.time()

    for name, sid_ in SUBJECTS.items():
        try:
            fetch_one(name, sid_, sid)
        except Exception as e:
            print(f"[ERROR] {name}: {e}", flush=True)

    elapsed = time.time() - start_time
    print(f"\n=== All done in {elapsed:.0f}s ===", flush=True)

    # Summary
    total = 0
    for f in sorted(Path(OUTPUT_DIR).glob("*.json")):
        try:
            count = len(json.loads(f.read_text(encoding="utf-8")))
            total += count
            print(f"  {f.name}: {count}", flush=True)
        except:
            pass
    print(f"  TOTAL: {total}", flush=True)


if __name__ == "__main__":
    main()
