# MemCube Political: 政治理论概念图谱扩增系统

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-orange.svg)](https://github.com/your-repo/memcube-political)

一个基于人工智能的政治理论概念图谱自动构建和扩增系统，通过深度学习和自然语言处理技术，从种子概念出发，迭代构建完整的政治理论知识图谱。

## ✨ 核心特性

### 🧠 智能概念扩增
- **基于LLM的概念推理**: 使用Gemini等大语言模型进行概念推理和扩增
- **多维度概念验证**: 语义相似度、概念质量、政治理论相关性、语言质量
- **自适应收敛控制**: 智能判断扩增收敛点，避免过度扩增

### 📊 多数据库支持
- **图数据库**: Neo4j、ArangoDB、JanusGraph支持
- **向量数据库**: Qdrant、ChromaDB、FAISS、Milvus集成
- **缓存优化**: 多级缓存机制，避免重复计算

### 🔍 高级分析功能
- **知识图谱可视化**: 支持多种图可视化格式
- **概念关系挖掘**: 深度挖掘概念间的语义关系
- **QA自动生成**: 基于概念图谱生成问答数据集

### ⚡ 高性能处理
- **并发处理**: 支持多线程并发扩增
- **批处理优化**: 高效的批量向量化操作
- **内存优化**: 智能内存管理和缓存策略

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Neo4j 4.0+ (可选)
- Qdrant 0.8+ (推荐)
- 8GB+ RAM
- 支持GPU的机器 (可选，用于加速)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/memcube-political.git
cd memcube-political

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置API密钥
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑 config/api_keys.yaml 填入你的API密钥

# 配置系统参数
cp config/config.yaml.example config/config.yaml
# 根据需要编辑配置文件
```

### 快速运行

```bash
# 运行完整的概念扩增流程
python main.py

# 或者分步骤运行
python -c "
from src.concept_graph import ConceptExpander
expander = ConceptExpander('config/config.yaml')
results = expander.run_full_expansion()
"
```

## 📖 详细文档

- [📚 用户手册](docs/USER_MANUAL.md) - 详细使用指南
- [🔧 安装指南](docs/INSTALLATION.md) - 完整安装说明
- [⚙️ 配置指南](docs/CONFIGURATION.md) - 配置文件详解
- [🔌 API参考](docs/API_REFERENCE.md) - 完整API文档
- [🏗️ 项目结构](docs/PROJECT_STRUCTURE.md) - 代码架构说明

## 🎯 使用场景

### 学术研究
- 政治理论概念体系构建
- 思想流派关系分析
- 理论演进路径研究

### 教育应用
- 政治理论知识图谱构建
- 智能问答系统开发
- 个性化学习路径设计

### 数据分析
- 文本概念抽取
- 主题关系挖掘
- 知识网络分析

## 🏗️ 系统架构

```
MemCube Political System
├── 数据输入层
│   ├── 种子概念输入
│   ├── 外部数据导入
│   └── 配置参数设置
├── 核心处理层
│   ├── 概念扩增引擎
│   ├── 向量化处理
│   ├── 图数据库操作
│   └── 向量数据库操作
├── 验证优化层
│   ├── 概念有效性验证
│   ├── 相似度计算
│   └── 收敛性判断
└── 输出展示层
    ├── 知识图谱导出
    ├── 可视化图表
    └── QA数据集生成
```

## 🔧 核心组件

### 1. ConceptExpander (`src/concept_graph.py`)
概念扩增的核心引擎，负责：
- 种子概念初始化
- 迭代概念扩增
- 概念验证和过滤
- 收敛性判断

### 2. EmbeddingClient (`src/embedding_client.py`)
向量化处理组件，支持：
- 多种embedding模型
- 本地Ollama模型
- 在线API模型
- 批量向量化优化

### 3. VectorDatabaseClient (`src/vector_database_client.py`)
向量数据库统一接口，支持：
- Qdrant (推荐)
- ChromaDB
- FAISS
- Milvus

### 4. GraphDatabaseClient (`src/graph_database_client.py`)
图数据库统一接口，支持：
- Neo4j (推荐)
- ArangoDB
- JanusGraph

## 📊 性能特性

### 扩增效率
- **单次迭代**: 可处理50-100个概念
- **并发扩增**: 支持多线程并行处理
- **缓存机制**: 90%+的重复计算避免率

### 存储优化
- **向量化**: 支持数百万概念的向量存储
- **图结构**: 高效的图查询和遍历
- **压缩存储**: 节省50%+的存储空间

### 可扩展性
- **水平扩展**: 支持分布式部署
- **模块化设计**: 易于集成新的数据库和模型
- **插件架构**: 支持自定义验证和处理逻辑

## 🔍 示例输出

### 概念图谱结构
```yaml
seed_concepts:
  - "马克思主义"
  - "社会主义"
  - "资本主义"

expanded_concepts:
  - "共产主义":
      relationships: ["马克思主义", "社会主义"]
      confidence: 0.95
  - "阶级斗争":
      relationships: ["马克思主义", "历史唯物主义"]
      confidence: 0.92
  - "剩余价值":
      relationships: ["马克思主义", "资本论"]
      confidence: 0.89
```

### QA生成示例
```json
{
  "question": "什么是马克思主义理论的核心概念？",
  "answer": "马克思主义理论的核心概念包括历史唯物主义、阶级斗争、剩余价值理论等。",
  "concepts": ["马克思主义", "历史唯物主义", "阶级斗争", "剩余价值"],
  "difficulty": "中等"
}
```

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与项目开发。

### 开发环境设置
```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/

# 代码格式化
black src/
flake8 src/
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 Google Gemini API 提供的强大语言模型支持
- 感谢 Neo4j 和 Qdrant 提供的优秀数据库产品
- 感谢开源社区的贡献者和用户

## 📞 联系我们

- 项目主页: https://github.com/your-repo/memcube-political
- 问题反馈: https://github.com/your-repo/memcube-political/issues
- 邮箱: your-email@example.com

---

**⭐ 如果这个项目对你有帮助，请给我们一个星标！**