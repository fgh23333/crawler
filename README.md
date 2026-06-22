# crawler

上海海洋大学马克思主义学院题库爬取与数据处理系统。

## 项目概述

本项目是一个完整的数据流水线系统，包含以下功能：

- **题库爬取**：从学校内部服务器抓取政治理论课考试题目
- **数据清洗**：去重、分类、格式化
- **数据合并**：新旧数据增量合并
- **试卷组装**：按题型均匀组卷
- **云端存储**：上传至 Cloudflare R2 并通过 Worker API 对外提供访问
- **知识图谱**：基于 MemCube 的概念抽取与图谱构建

## 涵盖科目

| 代码 | 科目名称 |
|------|---------|
| `Marx` | 马克思主义基本原理 |
| `CMH` | 中国近现代史纲要 |
| `Political` | 思想道德与法治 |
| `MaoIntro` | 毛泽东思想和中国特色社会主义理论体系概论 |
| `XiIntro` | 习近平新时代中国特色社会主义思想概论 |
| `SDH` | 社会主义发展史 |
| `NCH` | 新中国史 |
| `CCPH` | 中共党史 |
| `ORH` | 改革开放史 |

## 题型分类

| 类型 | 说明 |
|------|------|
| `singleChoice` | 单选题 |
| `multipleChoice` | 多选题 |
| `rightWrong` | 判断题 |
| `fillingBlank` | 填空题 |
| `subject` | 全部题目合集 |

## 项目结构

```
crawler/
├── scripts/
│   ├── pipeline.js              # 主流水线脚本（7步完整流程）
│   ├── pipeline-incremental.js  # 增量流水线（单科目处理）
│   ├── upload-r2.js             # R2 上传脚本
│   ├── data-fetch/              # 数据爬取
│   │   ├── new_index.js         # 新版爬虫（axios 并发）
│   │   ├── multiple.js          # 多线程爬虫（含重试）
│   │   └── main.js              # worker_threads 启动器
│   ├── data-process/            # 数据处理
│   │   ├── clear.js             # 去重
│   │   ├── merge.js             # 新旧数据合并
│   │   └── classify.js          # 题型分类
│   ├── data-transform/          # 数据转换
│   │   ├── data_transformer.js  # 转为标准化 QA 格式
│   │   └── concept_extractor.js # 政治概念抽取
│   └── utils/
│       └── test.js              # 原站 API 逆向参考
├── cloudflare-worker/
│   ├── index.js                 # Worker API 网关（CORS + 限流 + 缓存）
│   └── wrangler.toml            # Worker 部署配置
├── memcube-political/           # Python 知识图谱系统（MemCube）
├── memos/                       # 转换后的 QA 数据与种子概念
├── new/                         # 当前数据存储
│   ├── rewrite/                 # 分类后的题目（9科目 × 5类型）
│   ├── cura/                    # 组装试卷
│   └── solved/                  # 去重后的数据
├── merge/                       # 合并后的数据输出
└── .github/workflows/
    └── crawl.yml                # GitHub Actions 定时任务
```

## 数据流水线

```
爬取 → 去重 → 分类 → 合并 → 标注类型 → 组卷 → 上传 R2
```

## 快速开始

### 安装依赖

```bash
npm install
```

### 运行完整流水线

```bash
node scripts/pipeline.js
```

支持指定日期参数：

```bash
node scripts/pipeline.js --date=2026-06-01
```

### 运行增量流水线

```bash
# 处理全部科目
node scripts/pipeline-incremental.js

# 处理单个科目
node scripts/pipeline-incremental.js Marx
```

### 上传数据到 R2

```bash
CLOUDFLARE_ACCOUNT_ID=xxx CLOUDFLARE_API_TOKEN=xxx node scripts/upload-r2.js
```

## Cloudflare Worker API

Worker 部署后提供以下接口：

| 路径 | 说明 |
|------|------|
| `/rewrite/{subject}_{type}.json` | 获取分类后的题目数据 |
| `/version.json` | 获取数据版本信息 |

默认限流：60 次/分钟/IP。

## GitHub Actions 自动化

- **定时任务**：每周一北京时间 10:00 自动执行
- **手动触发**：支持 `workflow_dispatch` 手动运行
- **自动流程**：爬取 → 处理 → 提交数据 → 上传 R2 → 更新版本清单

### 必需 Secrets

在仓库 Settings → Secrets and variables → Actions 中配置：

| Secret 名 | 格式 | 说明 |
|----------|------|------|
| `STUDENT_NUMS` | JSON 数组字符串，如 `["2352613","2352614","2352615"]` | 抓取登录用的**学号**轮换池，密码=学号本身。启动时用每个学号调 `/login/practiceLogin` 登录拿 `studentId`，抓取时按 round-robin 轮换学号，避免单一身份被服务器限流；`studentId` 过期会自动重新登录。未设置或格式非法时回退到内置单一学号。 |
| `CLOUDFLARE_ACCOUNT_ID` | 字符串 | R2 上传账号 ID |
| `CLOUDFLARE_API_TOKEN` | 字符串 | R2 上传 API Token |

## 许可证

MIT
