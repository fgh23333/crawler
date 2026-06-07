# CRAWLER.md - Political Theory Question Bank & Knowledge Graph System SOP

## Overview

This CLI harness wraps a dual-purpose system:
1. **Node.js Crawler** - Fetches, processes, and classifies political theory exam questions from 9 subjects
2. **Python AI Engine** (MemCube Political) - Builds concept graphs, generates QA pairs using LLMs

## Supported Subjects

| Code  | Full Name                                          |
|-------|----------------------------------------------------|
| XiIntro | 习近平新时代中国特色社会主义思想概论           |
| Marx  | 马克思主义基本原理                                  |
| MaoIntro | 毛泽东思想和中国特色社会主义理论体系概论       |
| Political | 思想道德与法治                                     |
| CMH   | 中国近现代史纲要                                    |
| NCH   | 新中国史                                            |
| SDH   | 社会主义发展史                                      |
| ORH   | 改革开放史                                          |
| CCPH  | 中共党史                                            |

## Question Types

- `singleChoice` - Single choice (4 options, 1 answer)
- `multipleChoice` - Multiple choice (4+ options, multiple answers)
- `rightWrong` - True/False (2 options: 正确/错误)
- `fillingBlank` - Fill in the blank
- `subject` - Subjective/essay questions

## Data Pipeline

```
crawl fetch → data rewrite → data classify → data merge → data transform → concept extract → concept expand → qa generate
```

## Key File Locations

- Crawler scripts: `scripts/data-fetch/`, `scripts/data-process/`, `scripts/data-transform/`
- Python AI engine: `memcube-political/`
- Raw crawled data: `new/`, `2025-05-27/`
- Classified data: `2025-05-27/rewrite/`
- Merged data: `merge/`
- Transformed data: `memos/transformed_political_data.json`
- Seed concepts: `memos/political_seed_concepts.json`
- Concept graph output: `memcube-political/data/`
- Config: `memcube-political/config/config.yaml`, `memcube-political/config/api_keys.yaml`

## API Endpoint

- Base URL: `http://222.73.57.153:6571`
- Path: `/examinationInfo/getPracticeInfo`
- Method: POST
- Params: `branchId`, `chapterId`, `studentId`, `subjectId`
