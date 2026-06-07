"""Concept module - manage concept graphs, extraction, and expansion via MemCube Political."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .crawl import find_project_root


# Political concept patterns for seed extraction
CONCEPT_PATTERNS = {
    "marxism": [
        "马克思主义", "唯物主义", "辩证法", "历史唯物主义", "生产力", "生产关系",
        "经济基础", "上层建筑", "阶级斗争", "无产阶级", "资产阶级", "剩余价值",
        "资本", "商品", "价值", "使用价值", "交换价值", "劳动价值论", "剩余价值理论",
        "异化", "共产主义", "社会主义", "资本主义", "所有制", "公有制", "私有制",
        "社会形态", "生产方式", "劳动分工", "物质资料", "生产资料", "劳动资料",
        "劳动对象", "社会实践", "认识论", "真理", "实践", "感性认识", "理性认识",
        "意识", "物质", "运动", "时间", "空间", "矛盾", "对立统一", "量变", "质变",
        "否定之否定", "规律", "联系", "发展", "人民群众", "历史创造者", "社会存在",
        "社会意识", "文化", "意识形态", "国家", "政党", "民主", "法治", "自由",
        "平等", "正义", "公平", "效率", "改革", "革命", "创新",
    ],
    "mao_thought": [
        "毛泽东思想", "实事求是", "群众路线", "独立自主", "新民主主义革命",
        "社会主义改造", "社会主义建设", "人民民主专政", "统一战线", "武装斗争",
        "党的建设", "农村包围城市", "论持久战", "矛盾论", "实践论", "为人民服务",
    ],
    "xi_thought": [
        "习近平新时代中国特色社会主义思想", "中国梦", "两个一百年奋斗目标",
        "五位一体总体布局", "四个全面战略布局", "新发展理念", "高质量发展",
        "供给侧结构性改革", "全面深化改革", "全面依法治国", "全面从严治党",
        "人类命运共同体", "一带一路", "四个自信", "共同富裕", "乡村振兴",
    ],
}

# Regex suffix patterns for concept extraction
SUFFIX_PATTERNS = [
    "主义", "思想", "理论", "观念", "意识", "精神", "价值",
    "制度", "体制", "道路", "文明", "建设", "发展", "改革",
    "革命", "运动", "斗争", "阶级", "政党", "民主", "法治",
    "社会", "经济", "政治", "文化",
]


def extract_concepts(input_file: Optional[str] = None, output_dir: Optional[str] = None) -> dict:
    """Extract seed concepts from QA data using pattern matching.

    Uses the same logic as concept_extractor.js.

    Args:
        input_file: Input QA JSON file (transformed_political_data.json)
        output_dir: Output directory for concepts

    Returns:
        dict with status, concept count, sample concepts
    """
    import re

    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    in_file = Path(input_file) if input_file else root / "memos" / "transformed_political_data.json"
    out_dir = Path(output_dir) if output_dir else root / "memos"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        return {"status": "error", "message": f"Input file not found: {in_file}"}

    data = json.loads(in_file.read_text(encoding="utf-8"))
    concepts = set()

    # Add known patterns
    for group_patterns in CONCEPT_PATTERNS.values():
        for p in group_patterns:
            concepts.add(p)

    # Extract from QA data
    for item in data:
        text = item.get("question", "")
        for suffix in SUFFIX_PATTERNS:
            matches = re.findall(rf"[^，。！？\s]{{2,8}}{suffix}", text)
            for m in matches:
                if 2 <= len(m) <= 10:
                    concepts.add(m)

        for opt in item.get("options", []):
            if isinstance(opt, dict):
                text = opt.get("content", "")
            else:
                text = str(opt)
            for suffix in SUFFIX_PATTERNS:
                matches = re.findall(rf"[^，。！？\s]{{2,8}}{suffix}", text)
                for m in matches:
                    if 2 <= len(m) <= 10:
                        concepts.add(m)

    # Score and sort
    def score(c):
        l = len(c)
        if 4 <= l <= 8:
            return 10
        elif 2 <= l <= 10:
            return 5
        return 1

    cleaned = sorted(concepts, key=score, reverse=True)
    cleaned = [c for c in cleaned if c and 2 <= len(c) <= 20]

    # Save
    json_out = out_dir / "political_seed_concepts.json"
    txt_out = out_dir / "political_seed_concepts.txt"

    output_data = {
        "metadata": {
            "total_concepts": len(cleaned),
            "source_qa_count": len(data),
            "extraction_date": __import__("datetime").datetime.now().isoformat(),
            "description": "政治理论种子概念列表，用于概念图迭代扩增",
        },
        "seed_concepts": cleaned,
    }
    json_out.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_out.write_text("\n".join(cleaned), encoding="utf-8")

    return {
        "status": "done",
        "total_concepts": len(cleaned),
        "source_questions": len(data),
        "json_output": str(json_out),
        "txt_output": str(txt_out),
        "sample_concepts": cleaned[:20],
    }


def run_concept_expansion(config_file: Optional[str] = None, stage: str = "concept-expansion") -> dict:
    """Run MemCube Political concept expansion stages.

    Args:
        config_file: Path to config.yaml
        stage: Stage to run (concept-analysis, concept-extraction, concept-expansion)

    Returns:
        dict with status, output files
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    memcube_dir = root / "memcube-political"
    if not memcube_dir.exists():
        return {"status": "error", "message": f"MemCube directory not found: {memcube_dir}"}

    cfg = config_file or str(memcube_dir / "config" / "config.yaml")
    if not Path(cfg).exists():
        return {"status": "error", "message": f"Config file not found: {cfg}"}

    # Run via subprocess
    env = {**__import__("os").environ, "PYTHONPATH": str(memcube_dir / "src")}
    result = subprocess.run(
        [sys.executable, str(memcube_dir / "src" / "main.py"), "--stage", stage, "--config", cfg],
        capture_output=True, text=True, timeout=7200, cwd=str(memcube_dir), env=env,
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
        "stage": stage,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
    }


def graph_info(graph_file: Optional[str] = None) -> dict:
    """Show information about a concept graph file.

    Args:
        graph_file: Path to concept graph JSON file

    Returns:
        dict with status, graph statistics
    """
    root = find_project_root()
    if not root:
        return {"status": "error", "message": "Cannot find crawler project root"}

    search_dir = root / "memcube-political" / "data"

    if graph_file:
        gf = Path(graph_file)
    else:
        # Find the latest concept graph file
        candidates = list(search_dir.rglob("final_concept_graph.json"))
        if not candidates:
            candidates = list(search_dir.rglob("*concept_graph*.json"))
        if not candidates:
            return {"status": "error", "message": "No concept graph files found"}
        gf = max(candidates, key=lambda p: p.stat().st_mtime)

    if not gf.exists():
        return {"status": "error", "message": f"Graph file not found: {gf}"}

    data = json.loads(gf.read_text(encoding="utf-8"))

    # Analyze structure
    if isinstance(data, dict):
        nodes = data.get("nodes", data.get("concepts", []))
        edges = data.get("edges", data.get("relations", []))
    elif isinstance(data, list):
        nodes = data
        edges = []
    else:
        nodes = []
        edges = []

    return {
        "status": "done",
        "file": str(gf),
        "file_size_kb": round(gf.stat().st_size / 1024, 1),
        "node_count": len(nodes) if isinstance(nodes, list) else "unknown",
        "edge_count": len(edges) if isinstance(edges, list) else "unknown",
        "data_type": type(data).__name__,
    }
