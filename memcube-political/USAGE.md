# MemCube 政治理论概念图扩增系统使用指南

## 🎯 系统概述

MemCube 政治理论概念图扩增系统是基于 MemCube 框架构建的专门用于政治理论领域的知识图谱构建工具。系统通过以下四个主要阶段，从种子概念出发，构建完整的政治理论知识图谱并生成高质量的QA知识对。

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入项目目录
cd memcube-political

# 安装依赖
pip install -r requirements.txt

# 或使用快速启动脚本（自动检查依赖）
python quick_start.py
```

### 2. API配置

编辑 `config/api_keys.yaml` 文件，填入你的OpenAI API密钥：

```yaml
openai:
  api_key: "your-openai-api-key-here"
  organization: "your-organization-id-here"  # 可选
```

### 3. 运行系统

```bash
# 运行完整流程（推荐）
python -m src.main --stage all

# 或者使用快速启动脚本
python quick_start.py
```

## 📋 详细使用说明

### 支持的运行阶段

系统支持以下运行模式：

```bash
# 1. 完整流程（推荐）
python -m src.main --stage all

# 2. 仅概念图扩增
python -m src.main --stage concept-expansion

# 3. 仅QA生成（需要先有概念图）
python -m src.main --stage qa-generation
```

### 分阶段运行详解

#### 第一阶段：概念图扩增

```bash
python -m src.main --stage concept-expansion
```

**功能：**
- 基于种子概念进行迭代扩增
- 使用embedding去重保证概念质量
- 自动收敛控制避免无限扩增
- 支持并发处理提高效率

**输出文件：**
- `data/concept_graph/final_concept_graph.json` - 最终概念图
- `data/concept_graph/convergence_history.json` - 收敛历史
- `data/concept_graph/expansion_summary.json` - 扩增摘要

#### 第二阶段：QA生成

```bash
python -m src.main --stage qa-generation
```

**功能：**
- 为单个概念生成深度QA对
- 为概念对生成关联性QA对
- 质量控制和去重处理
- 支持多种问题类型和难度

**输出文件：**
- `results/political_theory_qa_dataset.json` - 完整QA数据集
- `results/political_theory_qa_training.jsonl` - 训练格式数据

## ⚙️ 配置说明

### 主配置文件：`config/config.yaml`

```yaml
# API配置
api:
  model_thinker: "gpt-4"           # 概念思考分析模型
  model_extractor: "gpt-4o-mini"    # 概念提取模型
  model_expander: "gpt-4o-mini"     # 概念扩增模型
  model_qa_generator: "gpt-4"       # QA生成模型

# 概念图扩增参数
concept_expansion:
  similarity_threshold: 0.80        # 概念相似度阈值
  new_concept_rate_threshold: 0.10   # 新概念增长率阈值
  new_edge_rate_threshold: 0.05      # 新边增长率阈值
  max_iterations: 10                 # 最大迭代次数
  max_workers: 10                    # 并发工作数

# QA生成参数
qa_generation:
  concepts_per_batch: 20            # 每批处理的概念数
  qa_pairs_per_concept: 3            # 每个概念生成的QA对数
  max_workers: 5                     # QA生成并发数

# Embedding配置
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 调优建议

**概念图扩增调优：**
- `similarity_threshold`: 降低以获得更多概念，提高以减少冗余
- `new_concept_rate_threshold`: 降低以获得更完整的图谱
- `max_workers`: 根据API限制调整

**QA生成调优：**
- `model_qa_generator`: 使用更强大的模型获得更高质量的QA
- `qa_pairs_per_concept`: 增加以获得更多训练数据

## 📊 输出格式

### 概念图格式

```json
{
  "graph": {
    "马克思主义": ["唯物主义", "辩证法", "历史唯物主义"],
    "唯物主义": ["马克思主义", "辩证唯物主义"],
    ...
  },
  "concept_embeddings": {
    "马克思主义": [0.1, 0.2, ...],
    "唯物主义": [0.3, 0.4, ...],
    ...
  },
  "metadata": {
    "total_iterations": 5,
    "final_nodes": 500,
    "final_edges": 1200
  }
}
```

### QA数据格式

```json
{
  "metadata": {
    "total_qa_pairs": 10000,
    "generation_model": "gpt-4",
    "timestamp": "2025-11-10T12:00:00"
  },
  "qa_pairs": [
    {
      "question": "马克思主义的基本特征是什么？",
      "answer": "马克思主义具有科学性、革命性、实践性、人民性等基本特征...",
      "difficulty": "medium",
      "type": "concept_understanding",
      "concept": "马克思主义",
      "source": "single_concept",
      "timestamp": "2025-11-10T12:00:00"
    },
    ...
  ]
}
```

## 🔍 质量评估

系统提供了完整的评估模块：

```python
# 运行评估
python -c "
from src.evaluation import evaluate_memcube_system
report = evaluate_memcube_system(
    graph_file='data/concept_graph/final_concept_graph.json',
    qa_file='results/political_theory_qa_dataset.json'
)
print(f'总体评分: {report.overall_score:.2f}')
print('建议:', report.recommendations)
"
```

### 评估指标

**概念图质量指标：**
- 结构完整性：连通性、密度、聚类系数
- 语义质量：概念多样性、相似度分布
- 覆盖度：概念覆盖范围、连通性

**QA质量指标：**
- 基础质量：问题长度、答案长度、格式正确性
- 内容多样性：问题类型分布、来源分布
- 概念覆盖度：概念在QA中的覆盖情况

## 🛠️ 高级用法

### 自定义种子概念

```python
# 准备自己的种子概念文件
echo "自定义概念1" > data/custom_seed_concepts.txt
echo "自定义概念2" >> data/custom_seed_concepts.txt

# 修改配置文件中的种子概念路径
# vim config/config.yaml
# paths:
#   seed_concepts: "data/custom_seed_concepts.txt"
```

### 批量处理

```python
# 使用Python API进行批量处理
from src.concept_graph import expand_concept_graph

# 扩展概念图
result_dir = expand_concept_graph(
    seed_concepts_file="data/seed_concepts.txt"
)

# 生成QA
from src.qa_generator import generate_political_theory_qa

qa_result = generate_political_theory_qa(
    concept_graph_file=f"{result_dir}/final_concept_graph.json"
)
```

### 自定义评估

```python
from src.evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator("config/config.yaml")
report = evaluator.evaluate_full_system(
    graph_file="path/to/graph.json",
    qa_file="path/to/qa.json"
)

# 获取详细评估结果
print(f"完整性评分: {report.completeness_score}")
print(f"质量评分: {report.quality_score}")
print(f"总体评分: {report.overall_score}")
```

## 🔧 故障排除

### 常见问题

**1. API密钥配置错误**
```
错误: 认证错误
解决: 检查config/api_keys.yaml中的API密钥是否正确
```

**2. 依赖包安装失败**
```
错误: No module named 'xxx'
解决: pip install -r requirements.txt
```

**3. 内存不足**
```
错误: CUDA out of memory
解决: 减少max_workers或batch_size配置
```

**4. 概念扩增不收敛**
```
解决:
- 降低similarity_threshold
- 增加max_iterations
- 检查种子概念质量
```

### 日志查看

```bash
# 查看最新日志
tail -f logs/memcube_$(date +%Y-%m-%d).log

# 查看错误日志
grep -i error logs/memcube_*.log
```

## 📈 性能优化

### 并发设置

```yaml
# 高性能配置（适合充足的API配额）
concept_expansion:
  max_workers: 20
  batch_size: 100

qa_generation:
  max_workers: 10
  concepts_per_batch: 50
```

```yaml
# 节省配置（适合有限的API配额）
concept_expansion:
  max_workers: 3
  batch_size: 10

qa_generation:
  max_workers: 2
  concepts_per_batch: 10
```

### 模型选择建议

**考虑因素：**
- 质量要求：GPT-4 > GPT-4o > GPT-4o-mini
- 成本考虑：GPT-4o-mini < GPT-4o < GPT-4
- 速度要求：GPT-4o-mini 最快

**推荐配置：**
- 概念分析：GPT-4（需要深度思考）
- 概念提取：GPT-4o-mini（结构化任务）
- 概念扩增：GPT-4o-mini（关联推理）
- QA生成：GPT-4（需要高质量输出）

## 📚 API参考

### 核心类

- `ConceptAnalyzer`: 概念分析器
- `ConceptExtractor`: 概念提取器
- `ConceptGraph`: 概念图构建器
- `QAGenerator`: QA生成器
- `ComprehensiveEvaluator`: 综合评估器

### 示例代码

```python
from src.concept_analyzer import ConceptAnalyzer
from src.concept_graph import ConceptGraph
from src.qa_generator import QAGenerator

# 1. 分析概念
analyzer = ConceptAnalyzer("config/config.yaml")
results = analyzer.analyze_concepts_batch(["马克思主义", "唯物主义"])

# 2. 构建概念图
graph = ConceptGraph(["马克思主义", "唯物主义"])
iteration_results = graph.run_full_expansion()

# 3. 生成QA
qa_generator = QAGenerator("config/config.yaml")
qa_result = qa_generator.run_full_qa_generation("path/to/graph.json")
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd memcube-political

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8

# 运行测试
pytest tests/

# 代码格式化
black src/
flake8 src/
```

## 📄 许可证

本项目基于MIT许可证开源。

## 🆘 支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 提交GitHub Issue
3. 联系开发团队

---

*最后更新：2025年11月10日*