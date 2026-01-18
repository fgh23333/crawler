# 配置指南 - MemCube Political

## 目录
1. [配置文件概述](#配置文件概述)
2. [API配置](#api配置)
3. [概念扩增配置](#概念扩增配置)
4. [向量化配置](#向量化配置)
5. [图数据库配置](#图数据库配置)
6. [向量数据库配置](#向量数据库配置)
7. [路径配置](#路径配置)
8. [日志配置](#日志配置)
9. [性能优化配置](#性能优化配置)
10. [环境变量配置](#环境变量配置)
11. [配置验证](#配置验证)

## 配置文件概述

MemCube Political 使用YAML格式的配置文件来管理所有系统参数。主要配置文件包括：

- `config/config.yaml` - 主配置文件
- `config/api_keys.yaml` - API密钥配置文件
- `.env` - 环境变量配置文件 (可选)

### 配置文件优先级

系统按照以下优先级读取配置：

1. 环境变量 (`.env` 文件或系统环境变量)
2. `config/api_keys.yaml` - API密钥
3. `config/config.yaml` - 主配置
4. 默认配置值 (代码中定义)

## API配置

### API模型配置

```yaml
api:
  # 核心模型配置
  model_thinker: "gemini-2.5-flash"      # 思考分析模型
  model_extractor: "gemini-2.5-flash"    # 概念提取模型
  model_expander: "gemini-2.5-flash"     # 概念扩增模型
  model_qa_generator: "gemini-2.5-flash" # QA生成模型

  # 通用API参数
  temperature: 0.7                        # 创造性: 0.0-2.0
  max_tokens: 32768                       # 最大生成token数
  max_retries: 3                          # 最大重试次数
  timeout: 60                             # 超时时间(秒)

  # 代理配置 (如需要)
  proxy:
    http: "http://127.0.0.1:7890"        # HTTP代理
    https: "http://127.0.0.1:7890"       # HTTPS代理
```

### 模型选择指南

| 模型 | 用途 | 特点 | 推荐场景 |
|------|------|------|----------|
| `gemini-2.5-flash` | 通用模型 | 速度快，成本低 | 概念扩增、QA生成 |
| `gemini-2.5-pro` | 高质量分析 | 质量高，成本中等 | 复杂概念分析 |
| `gpt-4` | 备用模型 | 通用性强 | Gemini不可用时 |
| `claude-3-sonnet` | 长文本处理 | 上下文窗口大 | 大批量处理 |

### API密钥配置

创建 `config/api_keys.yaml`:

```yaml
# Google Gemini API
gemini:
  api_key: "YOUR_GEMINI_API_KEY"
  # 获取密钥: https://makersuite.google.com/app/apikey
  base_url: "https://generativelanguage.googleapis.com"

# OpenAI API (备用)
openai:
  api_key: "YOUR_OPENAI_API_KEY"
  organization: "YOUR_ORG_ID"           # 可选
  base_url: "https://api.openai.com/v1"

# Anthropic Claude API (备用)
claude:
  api_key: "YOUR_CLAUDE_API_KEY"
  base_url: "https://api.anthropic.com"

# 智谱AI API (中文模型)
zhipuai:
  api_key: "YOUR_ZHIPU_API_KEY"
  base_url: "https://open.bigmodel.cn"

# 百度文心一言 (中文模型)
ernie:
  api_key: "YOUR_ERNIE_API_KEY"
  secret_key: "YOUR_ERNIE_SECRET_KEY"
  base_url: "https://aip.baidubce.com"
```

### API参数调优

#### 温度参数 (temperature)
```yaml
api:
  # 保守模式 - 高确定性
  temperature: 0.1  # 适合: 概念验证、标准答案生成

  # 平衡模式 - 适中的创造性
  temperature: 0.7  # 适合: 概念扩增、常规QA生成

  # 创新模式 - 高创造性
  temperature: 1.2  # 适合: 创新概念探索、创意QA

  # 随机模式 - 最高创造性
  temperature: 1.8  # 谨慎使用: 可能产生不稳定结果
```

#### Token配置
```yaml
api:
  # 短响应配置 - 快速处理
  max_tokens: 1000   # 适合: 简单概念验证

  # 中等响应配置 - 平衡速度和质量
  max_tokens: 4000   # 适合: 概念扩增

  # 长响应配置 - 高质量输出
  max_tokens: 8000   # 适合: 复杂概念分析

  # 超长响应配置 - 最大质量
  max_tokens: 32768  # 适合: 详细QA生成
```

## 概念扩增配置

### 基础扩增配置

```yaml
concept_expansion:
  # 收敛控制参数
  similarity_threshold: 0.80              # 概念相似度阈值 (0.0-1.0)
  new_concept_rate_threshold: 0.10        # 新概念增长率阈值
  new_edge_rate_threshold: 0.05           # 新边增长率阈值
  max_iterations: 10                      # 最大迭代次数

  # 批处理配置
  batch_size: 50                          # 每批处理的概念数
  max_workers: 10                         # 并发工作线程数

  # 概念验证配置
  validity_threshold: 0.6                 # 概念有效性阈值 (0.0-1.0)
  max_new_concepts_per_center: 20         # 每个中心概念最大新概念数
  min_concept_length: 2                   # 最小概念长度
  max_concept_length: 50                  # 最大概念长度

  # 高级参数
  enable_semantic_filtering: true         # 启用语义过滤
  enable_duplicate_detection: true        # 启用重复检测
  custom_validation_rules: []             # 自定义验证规则
```

### 扩增策略配置

#### 保守扩增策略
```yaml
concept_expansion:
  similarity_threshold: 0.85              # 高相似度阈值
  validity_threshold: 0.8                 # 高有效性阈值
  max_new_concepts_per_center: 10         # 限制新概念数量
  max_iterations: 5                       # 较少迭代次数
  batch_size: 25                          # 小批量处理
```

#### 激进扩增策略
```yaml
concept_expansion:
  similarity_threshold: 0.70              # 低相似度阈值
  validity_threshold: 0.5                 # 低有效性阈值
  max_new_concepts_per_center: 50         # 允许更多新概念
  max_iterations: 20                      # 更多迭代次数
  batch_size: 100                         # 大批量处理
```

#### 平衡扩增策略 (推荐)
```yaml
concept_expansion:
  similarity_threshold: 0.80              # 中等相似度阈值
  validity_threshold: 0.6                 # 中等有效性阈值
  max_new_concepts_per_center: 30         # 适中的新概念数量
  max_iterations: 10                      # 适中的迭代次数
  batch_size: 50                          # 中等批量
```

### 验证规则配置

```yaml
concept_expansion:
  # 概念质量检查权重
  validation_weights:
    semantic_similarity: 0.25             # 语义相似度权重
    concept_quality: 0.25                 # 概念质量权重
    political_theory_relevance: 0.30      # 政治理论相关性权重
    linguistic_quality: 0.20              # 语言质量权重

  # 概念质量标准
  quality_criteria:
    # 语义相似度检查
    min_semantic_similarity: 0.6          # 最小语义相似度

    # 概念长度检查
    min_concept_length: 2                 # 最小概念长度
    max_concept_length: 50                # 最大概念长度

    # 字符类型检查
    require_chinese_chars: true           # 要求包含中文字符
    max_special_chars_ratio: 0.2          # 最大特殊字符比例

    # 政治理论相关性
    political_keywords:                   # 政治理论关键词列表
      - "政治"
      - "经济"
      - "社会"
      - "文化"
      - "理论"
      - "思想"
      - "制度"
      - "民主"
      - "自由"
      - "平等"

    # 禁用词列表
    forbidden_words:                      # 禁用的词汇
      - "测试"
      - "示例"
      - "demo"
      - "test"
```

## 向量化配置

### Embedding模型配置

```yaml
embedding:
  # 模型选择
  model_name: "bge-m3:567m"              # 模型名称
  model_type: "ollama"                   # 模型类型: ollama, huggingface, openai

  # Ollama配置 (本地模型)
  ollama_url: "http://localhost:11434"   # Ollama服务地址
  ollama_timeout: 60                     # Ollama超时时间

  # HuggingFace配置 (在线模型)
  huggingface:
    model_name: "BAAI/bge-m3"
    device: "auto"                       # 设备选择: auto, cpu, cuda
    trust_remote_code: true              # 是否信任远程代码

  # OpenAI配置 (API模型)
  openai:
    model_name: "text-embedding-ada-002"
    api_base: "https://api.openai.com/v1"

  # 批处理配置
  batch_size: 16                         # 批处理大小
  normalize_embeddings: true             # 是否归一化向量

  # 性能配置
  device: "cpu"                          # 计算设备: cpu, cuda, mps
  max_sequence_length: 512               # 最大序列长度
  use_fp16: false                        # 是否使用半精度
```

### 模型选择指南

#### 本地模型 (Ollama)
```yaml
embedding:
  model_type: "ollama"
  model_name: "bge-m3:567m"              # 推荐: 多语言支持，性能好
  # model_name: "nomic-embed-text"       # 轻量级选择
  # model_name: "mxbai-embed-large"      # 高质量选择
```

#### 在线模型 (HuggingFace)
```yaml
embedding:
  model_type: "huggingface"
  model_name: "BAAI/bge-m3"              # 多语言模型
  # model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  # model_name: "shibing624/text2vec-base-chinese"  # 中文专用
```

#### API模型 (OpenAI)
```yaml
embedding:
  model_type: "openai"
  model_name: "text-embedding-3-small"    # 成本效益高
  # model_name: "text-embedding-3-large" # 高质量
  # model_name: "text-embedding-ada-002" # 经典选择
```

### 性能优化配置

#### CPU优化
```yaml
embedding:
  device: "cpu"
  batch_size: 32                         # 增大批处理
  num_threads: 4                         # CPU线程数
  use_fp16: false                        # CPU不支持fp16
```

#### GPU优化
```yaml
embedding:
  device: "cuda"
  batch_size: 128                        # GPU可以处理更大批量
  use_fp16: true                         # 启用半精度，节省内存
  max_sequence_length: 1024              # GPU支持更长序列
```

#### 内存优化
```yaml
embedding:
  batch_size: 8                          # 小批量，节省内存
  max_sequence_length: 256               # 限制序列长度
  device: "cpu"                          # CPU比GPU省内存
  use_quantization: true                 # 启用量化 (如支持)
```

## 图数据库配置

### Neo4j配置

```yaml
graph_database:
  enabled: true                          # 是否启用图数据库
  type: "neo4j"                          # 数据库类型

  # Neo4j连接配置
  neo4j:
    uri: "bolt://localhost:7687"         # 连接URI
    username: "neo4j"                     # 用户名
    password: "YOUR_NEO4J_PASSWORD"       # 密码
    database: "neo4j"                     # 数据库名称

    # 连接池配置
    max_connection_lifetime: 1800         # 连接最大生命周期(秒)
    max_connection_pool_size: 20          # 连接池大小
    connection_acquisition_timeout: 60    # 连接获取超时(秒)
    max_transaction_retry_time: 30        # 事务最大重试时间(秒)

    # 性能配置
    batch_size: 50                        # 批量写入大小
    batch_timeout: 60                     # 批量操作超时(秒)
    retry_attempts: 3                     # 重试次数
    retry_delay: 1.0                      # 重试延迟(秒)
```

### Neo4j服务器配置建议

#### 内存配置
```conf
# conf/neo4j.conf
# 堆内存设置
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G

# 页面缓存设置
server.memory.pagecache.size=2G

# 查询内存设置
server.memory.transaction.global_max_size=1G
```

#### 连接配置
```conf
# 最大连接数
server.bolt.thread_pool_min_size=5
server.bolt.thread_pool_max_size=400

# 连接超时
server.bolt.connection_keep_alive=30s
server.bolt.connection_timeout=60s
```

### 其他图数据库配置

#### ArangoDB配置
```yaml
graph_database:
  type: "arangodb"
  arangodb:
    host: "localhost"
    port: 8529
    username: "root"
    password: "YOUR_PASSWORD"
    database: "memcube_political"
    # 连接池配置
    pool_size: 10
    timeout: 30
```

#### JanusGraph配置
```yaml
graph_database:
  type: "janusgraph"
  janusgraph:
    host: "localhost"
    port: 8182
    graph_name: "political_concepts"
    storage_backend: "cql"                # 存储后端
    storage_hostname: "localhost"         # 存储主机
    storage_port: 9042                    # 存储端口
    storage_username: "janusgraph"        # 存储用户名
    storage_password: "YOUR_PASSWORD"     # 存储密码
```

## 向量数据库配置

### Qdrant配置 (推荐)

```yaml
vector_database:
  enabled: true                          # 是否启用向量数据库
  type: "qdrant"                          # 数据库类型

  # Qdrant连接配置
  qdrant:
    host: "localhost"                     # 主机地址
    port: 6333                           # HTTP端口
    grpc_port: 6334                      # gRPC端口
    api_key: null                        # API密钥 (如需要)
    collection_name: "political_concepts" # 集合名称

    # 向量配置
    vector_size: 1024                    # 向量维度
    distance: "Cosine"                   # 距离算法: Cosine, Euclidean, Dot

    # 性能配置
    search_top_k: 10                     # 搜索返回数量
    batch_size: 100                      # 批量插入大小
    similarity_threshold: 0.7            # 相似度阈值

    # 索引配置
    index_type: "HNSW"                   # 索引类型: HNSW, IVF, Flat
    hnsw_config:
      m: 16                               # HNSW M参数
      ef_construct: 200                  # HNSW 构建参数
      ef_search: 64                      # HNSW 搜索参数
```

### Qdrant性能优化

#### 内存优化
```yaml
vector_database:
  qdrant:
    # 减少内存使用
    vector_size: 768                     # 使用较小的向量维度
    batch_size: 50                       # 小批量
    hnsw_config:
      m: 8                               # 减少M值
      ef_construct: 100                  # 减少ef值
```

#### 速度优化
```yaml
vector_database:
  qdrant:
    # 提高查询速度
    search_top_k: 20                     # 增加返回数量
    hnsw_config:
      ef_search: 128                     # 增加ef值
    # 使用GPU (如支持)
    prefer_gpu: true
```

### 其他向量数据库配置

#### ChromaDB配置
```yaml
vector_database:
  type: "chroma"
  chroma:
    path: "./data/vector_db"             # 数据库路径
    collection_name: "political_concepts"
    persist_directory: "./data/vector_db"
    # 性能配置
    batch_size: 100
    search_top_k: 10
```

#### FAISS配置
```yaml
vector_database:
  type: "faiss"
  faiss:
    index_type: "IVF_PQ"                 # 索引类型
    dimension: 1024                      # 向量维度
    index_path: "./data/faiss_index"
    save_interval: 100                   # 保存间隔
    # IVF参数
    nlist: 100                           # 聚类中心数
    nprobe: 10                           # 搜索时的聚类数
    # PQ参数
    m: 64                                # PQ子向量数
    nbits: 8                             # 每个子向量位数
```

#### Milvus配置
```yaml
vector_database:
  type: "milvus"
  milvus:
    host: "localhost"
    port: 19530
    collection_name: "political_concepts"
    vector_size: 1024
    index_type: "IVF_FLAT"
    metric_type: "IP"                    # 内积距离
    nlist: 16384                         # 索引参数
```

## 路径配置

### 目录结构配置

```yaml
paths:
  # 输入数据路径
  seed_concepts: "data/seed_concepts.txt"           # 种子概念文件
  seed_concepts_json: "data/seed_concepts.json"     # 种子概念JSON文件
  qa_data: "data/transformed_political_data.json"   # QA数据文件
  knowledge_base: "data/political_theory_knowledge_base.yaml"  # 知识库文件

  # 输出数据路径
  concept_graph_dir: "data/concept_graph"           # 概念图谱目录
  results_dir: "results"                           # 结果输出目录
  export_dir: "results/exports"                    # 导出文件目录

  # 系统路径
  logs_dir: "logs"                                 # 日志目录
  scripts_dir: "scripts"                           # 脚本目录
  cache_dir: "data/cache"                          # 缓存目录
  temp_dir: "data/temp"                            # 临时文件目录

  # 模型路径
  model_cache_dir: "data/models"                   # 模型缓存目录
  embedding_cache_dir: "data/embeddings"           # embedding缓存目录
```

### 自动创建目录

系统会在启动时自动创建必要的目录。如需自定义路径：

```yaml
paths:
  # 自定义路径示例
  seed_concepts: "/your/custom/path/concepts.txt"
  results_dir: "/your/custom/path/results"
  logs_dir: "/your/custom/path/logs"
```

## 日志配置

### 日志级别和格式

```yaml
logging:
  level: "INFO"                          # 日志级别: DEBUG, INFO, WARNING, ERROR, CRITICAL

  # 日志格式
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

  # 日志轮转
  rotation: "1 day"                      # 轮转间隔
  retention: "30 days"                   # 保留时间

  # 文件配置
  log_file: "logs/concept_expansion.log" # 主日志文件
  error_file: "logs/errors.log"          # 错误日志文件
  debug_file: "logs/debug.log"           # 调试日志文件

  # 控制台输出
  console_output: true                   # 是否输出到控制台
  console_level: "INFO"                  # 控制台日志级别
```

### 详细日志配置

```yaml
logging:
  # 按模块配置日志级别
  modules:
    concept_graph: "INFO"                # 概念图谱模块
    embedding_client: "WARNING"          # Embedding客户端
    graph_database: "INFO"               # 图数据库
    vector_database: "INFO"              # 向量数据库
    api_client: "WARNING"                # API客户端

  # 特定功能的日志控制
  features:
    log_api_calls: false                 # 是否记录API调用详情
    log_embedding_cache: true            # 是否记录embedding缓存
    log_database_operations: true        # 是否记录数据库操作
    log_performance_metrics: true        # 是否记录性能指标

  # 性能日志
  performance:
    log_slow_queries: true               # 记录慢查询
    slow_query_threshold: 5.0            # 慢查询阈值(秒)
    log_memory_usage: true               # 记录内存使用
    memory_check_interval: 60            # 内存检查间隔(秒)
```

## 性能优化配置

### 内存优化

```yaml
# 大规模数据处理配置
performance:
  # 内存管理
  max_memory_usage: "8GB"                # 最大内存使用量
  memory_check_interval: 30              # 内存检查间隔(秒)
  enable_memory_monitoring: true         # 启用内存监控

  # 缓存配置
  enable_embedding_cache: true           # 启用embedding缓存
  cache_size_limit: "2GB"                # 缓存大小限制
  cache_cleanup_interval: 300            # 缓存清理间隔(秒)

  # 批处理优化
  auto_batch_size: true                  # 自动调整批量大小
  min_batch_size: 10                     # 最小批量大小
  max_batch_size: 200                    # 最大批量大小

  # 并发控制
  max_concurrent_tasks: 4                # 最大并发任务数
  worker_thread_pool_size: 8             # 工作线程池大小
```

### CPU优化

```yaml
performance:
  # CPU配置
  cpu_cores: 4                           # CPU核心数 (自动检测如为0)
  use_multiprocessing: true              # 启用多进程处理

  # 进程池配置
  process_pool_size: 4                   # 进程池大小
  max_tasks_per_worker: 100              # 每个worker的最大任务数

  # 任务调度
  task_queue_size: 1000                  # 任务队列大小
  task_timeout: 300                      # 任务超时时间(秒)
```

### GPU优化

```yaml
performance:
  # GPU配置
  use_gpu: true                          # 是否使用GPU
  gpu_memory_fraction: 0.8               # GPU内存使用比例
  gpu_device_id: 0                       # GPU设备ID

  # 混合精度训练
  use_mixed_precision: true              # 使用混合精度
  fp16_opt_level: "O1"                   # FP16优化级别

  # GPU内存管理
  enable_gradient_checkpointing: true    # 启用梯度检查点
  max_gpu_batch_size: 128                # GPU最大批量大小
```

## 环境变量配置

### .env文件配置

```bash
# .env 文件示例

# API密钥
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_HOST=localhost
QDRANT_PORT=6333

# 模型配置
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3:567m

# 路径配置
PROJECT_ROOT=/path/to/memcube-political
DATA_DIR=${PROJECT_ROOT}/data
RESULTS_DIR=${PROJECT_ROOT}/results
LOGS_DIR=${PROJECT_ROOT}/logs

# 性能配置
MAX_WORKERS=8
BATCH_SIZE=50
MEMORY_LIMIT=8GB

# 调试配置
DEBUG_MODE=false
LOG_LEVEL=INFO
```

### 系统环境变量

```bash
# 在系统中设置环境变量

# Linux/macOS
export GEMINI_API_KEY="your_key_here"
export NEO4J_PASSWORD="your_password"

# Windows
set GEMINI_API_KEY=your_key_here
set NEO4J_PASSWORD=your_password

# 或者在 ~/.bashrc 或 ~/.zshrc 中添加
echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc
source ~/.bashrc
```

## 配置验证

### 创建配置验证脚本

创建 `validate_config.py`:

```python
#!/usr/bin/env python3
"""
配置文件验证脚本
"""

import yaml
import os
from pathlib import Path

def validate_config():
    """验证配置文件"""
    config_file = "config/config.yaml"
    api_keys_file = "config/api_keys.yaml"

    issues = []

    # 检查配置文件存在性
    if not os.path.exists(config_file):
        issues.append(f"配置文件不存在: {config_file}")
    else:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                print("✓ 主配置文件格式正确")
        except yaml.YAMLError as e:
            issues.append(f"配置文件格式错误: {e}")

    # 检查API密钥文件
    if not os.path.exists(api_keys_file):
        issues.append(f"API密钥文件不存在: {api_keys_file}")
    else:
        try:
            with open(api_keys_file, 'r', encoding='utf-8') as f:
                api_keys = yaml.safe_load(f)
                print("✓ API密钥文件格式正确")

                # 检查必要的API密钥
                if 'gemini' not in api_keys or not api_keys['gemini'].get('api_key'):
                    issues.append("缺少Gemini API密钥")

        except yaml.YAMLError as e:
            issues.append(f"API密钥文件格式错误: {e}")

    # 检查目录结构
    directories = ["data", "logs", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ 目录存在或已创建: {directory}")

    # 验证配置值
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

            # 检查必需的配置项
            required_keys = [
                'api', 'concept_expansion', 'embedding',
                'graph_database', 'vector_database', 'paths'
            ]

            for key in required_keys:
                if key not in config:
                    issues.append(f"缺少必需配置: {key}")

    return issues

def main():
    print("=== 配置文件验证 ===")

    issues = validate_config()

    if not issues:
        print("\n🎉 配置验证通过！")
        print("系统配置正确，可以开始运行。")
        return True
    else:
        print(f"\n❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")

        print("\n请修复上述问题后重新运行验证。")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
```

### 运行配置验证

```bash
# 确保虚拟环境已激活
python validate_config.py
```

### 配置最佳实践

1. **备份配置**: 在修改配置前先备份原文件
2. **渐进式调整**: 一次只修改少量参数，测试效果
3. **性能监控**: 使用日志监控系统性能表现
4. **版本控制**: 将配置文件纳入版本控制 (排除敏感信息)
5. **环境隔离**: 为不同环境使用不同的配置文件

通过遵循本配置指南，您可以有效地配置和优化MemCube Political系统，以获得最佳的性能和稳定性。