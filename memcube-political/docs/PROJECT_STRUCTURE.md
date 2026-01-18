# 项目结构说明 - MemCube Political

## 目录
1. [总体架构](#总体架构)
2. [目录结构](#目录结构)
3. [核心模块详解](#核心模块详解)
4. [数据流架构](#数据流架构)
5. [模块依赖关系](#模块依赖关系)
6. [设计模式](#设计模式)
7. [扩展指南](#扩展指南)

## 总体架构

MemCube Political 采用模块化、分层的架构设计，将系统划分为多个独立但相互协作的组件。

### 架构层次

```
┌─────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)              │
├─────────────────────────────────────────────────────────┤
│                    服务层 (Service Layer)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  概念扩增服务     │  │  QA生成服务      │  │  分析服务    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    核心层 (Core Layer)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  概念图管理器     │  │  向量化引擎      │  │  验证引擎    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    数据层 (Data Layer)                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  图数据库客户端   │  │  向量数据库客户端 │  │  文件系统    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    基础设施层 (Infrastructure Layer)         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  API客户端       │  │  配置管理器      │  │  日志系统    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 核心原则

1. **模块化**: 每个功能模块独立开发和测试
2. **可扩展性**: 支持新数据库、模型和算法的集成
3. **配置驱动**: 通过配置文件控制系统行为
4. **错误容错**: 完善的错误处理和恢复机制
5. **性能优化**: 支持批处理、并发和缓存

## 目录结构

```
memcube-political/
├── README.md                    # 项目主文档
├── requirements.txt             # Python依赖列表
├── setup.py                    # 安装脚本
├── main.py                     # 主入口文件
├── run.py                      # 运行脚本
│
├── config/                     # 配置文件目录
│   ├── config.yaml            # 主配置文件
│   ├── config.yaml.example    # 配置文件模板
│   ├── api_keys.yaml          # API密钥配置
│   ├── api_keys.yaml.example  # API密钥模板
│   └── logging.yaml           # 日志配置
│
├── src/                       # 源代码目录
│   ├── __init__.py
│   ├── concept_graph.py       # 概念图谱扩增核心
│   ├── concept_analyzer.py    # 概念分析器
│   ├── concept_extractor.py   # 概念提取器
│   ├── concept_graph.py       # 概念图谱管理
│   ├── embedding_client.py    # 文本向量化客户端
│   ├── api_client.py          # API调用客户端
│   ├── graph_database_client.py    # 图数据库客户端
│   ├── vector_database_client.py   # 向量数据库客户端
│   ├── qa_generator.py        # QA数据生成器
│   ├── evaluation.py          # 评估模块
│   ├── templates.py           # 提示词模板
│   ├── knowledge_base_builder.py   # 知识库构建器
│   └── utils/                 # 工具函数目录
│       ├── __init__.py
│       ├── file_utils.py      # 文件操作工具
│       ├── text_utils.py      # 文本处理工具
│       ├── graph_utils.py     # 图处理工具
│       ├── validation_utils.py # 验证工具
│       └── config_utils.py    # 配置工具
│
├── data/                      # 数据目录
│   ├── seed_concepts.txt      # 种子概念文件
│   ├── seed_concepts.json     # 种子概念JSON
│   ├── transformed_political_data.json  # 转换后的政治数据
│   ├── political_theory_knowledge_base.yaml  # 政治理论知识库
│   ├── concept_graph/         # 概念图谱数据
│   ├── cache/                 # 缓存目录
│   ├── models/                # 模型缓存
│   ├── embeddings/            # 向量缓存
│   └── temp/                  # 临时文件
│
├── docs/                      # 文档目录
│   ├── USER_MANUAL.md         # 用户手册
│   ├── INSTALLATION.md        # 安装指南
│   ├── CONFIGURATION.md       # 配置指南
│   ├── API_REFERENCE.md       # API参考
│   ├── PROJECT_STRUCTURE.md   # 项目结构说明
│   └── images/                # 文档图片
│
├── scripts/                   # 脚本目录
│   ├── data-process/          # 数据处理脚本
│   │   ├── clean_data.py      # 数据清洗
│   │   ├── transform_data.py  # 数据转换
│   │   └── validate_data.py   # 数据验证
│   ├── data-transform/        # 数据转换脚本
│   │   ├── concept_extractor.py
│   │   └── qa_converter.py
│   ├── utils/                 # 通用脚本
│   │   ├── setup_env.py       # 环境设置
│   │   ├── test_connections.py # 连接测试
│   │   └── benchmark.py       # 性能测试
│   └── deployment/            # 部署脚本
│       ├── docker/            # Docker相关
│       └── kubernetes/        # K8s相关
│
├── tests/                     # 测试目录
│   ├── __init__.py
│   ├── test_concept_graph.py  # 概念图谱测试
│   ├── test_embedding_client.py  # 向量化测试
│   ├── test_databases.py      # 数据库测试
│   ├── test_qa_generator.py   # QA生成测试
│   ├── integration/           # 集成测试
│   └── fixtures/              # 测试数据
│
├── results/                   # 结果输出目录
│   ├── concept_graph.json     # 概念图谱JSON
│   ├── concept_graph.graphml  # 概念图谱GraphML
│   ├── qa_dataset.json        # QA数据集
│   ├── metrics/               # 评估指标
│   └── exports/               # 导出文件
│
├── logs/                      # 日志目录
│   ├── concept_expansion.log  # 概念扩增日志
│   ├── api_calls.log          # API调用日志
│   ├── database_operations.log # 数据库操作日志
│   └── performance.log        # 性能日志
│
├── .env.example               # 环境变量示例
├── .gitignore                 # Git忽略文件
├── Dockerfile                 # Docker文件
├── docker-compose.yml         # Docker Compose配置
├── requirements-dev.txt       # 开发依赖
└── pyproject.toml            # 项目配置
```

## 核心模块详解

### 1. 概念扩增核心 (`src/concept_graph.py`)

#### 主要类

**ConceptExpander**: 概念扩增的主要引擎

```python
class ConceptExpander:
    """概念扩增器"""

    def __init__(self, config_path: str)
    def load_seed_concepts(self, concepts_file: str)
    def run_full_expansion(self) -> List[Dict]
    def run_expansion_iteration(self) -> Dict
    def expand_single_concept(self, concept: str, neighbors: List[str])
    def check_convergence(self) -> Dict
    def export_graph_json(self, output_file: str)
```

**核心功能**:
- 种子概念管理
- 迭代概念扩增
- 概念验证和过滤
- 收敛性判断
- 图数据导出

#### 数据流

```
种子概念 → 概念扩增 → 概念验证 → 图更新 → 收敛检查 → 导出结果
    ↑                                                      ↓
    ←←←←←←←←←←←←←←←←←←←←← 迭代循环 ←←←←←←←←←←←←←←←←←←←←←←←←←
```

### 2. 向量化客户端 (`src/embedding_client.py`)

#### 主要类

**EmbeddingClient**: 统一的向量化接口

```python
class EmbeddingClient:
    """文本向量化客户端"""

    def __init__(self, config: Dict)
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray
    def similarity(self, text1: Union[str, np.ndarray], text2: Union[str, np.ndarray]) -> float
    def most_similar(self, query: str, candidates: List[str]) -> List[Tuple]
```

**支持的模型**:
- Ollama本地模型 (bge-m3, nomic-embed-text等)
- HuggingFace在线模型
- OpenAI API模型
- 本地自定义模型

#### 缓存机制

```python
class EmbeddingCache:
    """向量化缓存"""

    def get(self, text: str) -> Optional[np.ndarray]
    def set(self, text: str, vector: np.ndarray)
    def clear(self)
    def get_stats(self) -> Dict
```

### 3. API客户端 (`src/api_client.py`)

#### 主要类

**APIClient**: 统一的API调用接口

```python
class APIClient:
    """API调用客户端"""

    def chat_completion(self, messages: List[Dict], model: str) -> APIResponse
    def json_completion(self, messages: List[Dict], model: str) -> APIResponse
    def batch_completion(self, message_batches: List[List[Dict]]) -> List[APIResponse]
```

**支持的API**:
- Google Gemini API
- OpenAI API
- Anthropic Claude API
- 智谱AI API
- 百度文心一言

#### 重试和限流机制

```python
class RetryManager:
    """重试管理器"""

    def __init__(self, max_retries: int, backoff_factor: float)
    def execute_with_retry(self, func: Callable, *args, **kwargs)

class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests: int, time_window: int)
    def acquire(self) -> bool
```

### 4. 图数据库客户端 (`src/graph_database_client.py`)

#### 架构设计

```python
class GraphDatabaseClient(ABC):
    """图数据库抽象基类"""

    @abstractmethod
    def connect(self) -> bool

    @abstractmethod
    def add_node(self, node_id: str, labels: List[str], properties: Dict) -> bool

    @abstractmethod
    def add_edge(self, source: str, target: str, relationship: str, properties: Dict) -> bool

    @abstractmethod
    def query(self, query: str, parameters: Dict = None) -> List[Dict]

class Neo4jClient(GraphDatabaseClient):
    """Neo4j客户端实现"""

    def __init__(self, config: Dict)
    def batch_operations(self, operations: List[Dict]) -> Dict
    def test_connection(self) -> bool
```

**支持的数据库**:
- Neo4j (推荐)
- ArangoDB
- JanusGraph

### 5. 向量数据库客户端 (`src/vector_database_client.py`)

#### 架构设计

```python
class VectorDatabaseClient(ABC):
    """向量数据库抽象基类"""

    @abstractmethod
    def create_collection(self, name: str, vector_size: int) -> bool

    @abstractmethod
    def index_vectors(self, collection: str, vectors: List[Dict]) -> bool

    @abstractmethod
    def search(self, collection: str, query_vector: np.ndarray) -> List[Dict]

class QdrantClient(VectorDatabaseClient):
    """Qdrant客户端实现"""

    def check_concepts_exist(self, collection: str, concept_ids: List[str]) -> Dict
    def batch_search(self, collection: str, query_vectors: List[np.ndarray]) -> List
```

**支持的数据库**:
- Qdrant (推荐)
- ChromaDB
- FAISS
- Milvus

### 6. QA生成器 (`src/qa_generator.py`)

#### 主要类

```python
class QAGenerator:
    """QA数据生成器"""

    def __init__(self, config_path: str)
    def generate_qa_from_graph(self, graph: nx.Graph) -> List[Dict]
    def generate_concept_based_qa(self, concepts: List[str]) -> List[Dict]
    def generate_relationship_based_qa(self, relationships: List[Tuple]) -> List[Dict]
    def export_qa_pairs(self, qa_data: List[Dict], output_file: str)
```

**生成策略**:
- 基于概念的QA生成
- 基于关系的QA生成
- 基于路径的QA生成
- 基于子图的QA生成

## 数据流架构

### 1. 概念扩增流程

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   种子概念输入    │ → │   概念扩增API    │ → │   概念验证       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   结果导出       │ ← │   图更新        │ ← │   向量化        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. 数据存储架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   内存缓存       │    │   图数据库       │    │   向量数据库     │
│                 │    │   (Neo4j)       │    │   (Qdrant)      │
│ • 概念embedding  │    │ • 概念节点       │    │ • 概念向量       │
│ • 验证结果       │    │ • 关系边         │    │ • 相似度索引     │
│ • 统计信息       │    │ • 属性信息       │    │ • 元数据        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ↓                       ↓                       ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文件系统       │    │   批量操作       │    │   相似度搜索     │
│                 │    │   事务处理       │    │   批量查询       │
│ • 导出文件       │    │   连接池管理     │    │   索引优化       │
│ • 日志文件       │    │   重试机制       │    │   缓存策略       │
│ • 临时文件       │    │   错误恢复       │    │   数据持久化     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 3. 处理流水线

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   任务调度       │    │   批处理         │    │   并发控制       │
│                 │    │                 │    │                 │
│ • 任务队列       │    │ • 批量大小优化   │    │ • 线程池管理     │
│ • 优先级调度     │    │ • 内存管理       │    │ • 负载均衡       │
│ • 失败重试       │    │ • 超时控制       │    │ • 资源限制       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 模块依赖关系

### 依赖图

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.py                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  concept_graph.py                               │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐│
│  │ concept_analyzer│ concept_graph   │   concept_extractor     ││
│  └─────────────────┴─────────────────┴─────────────────────────┘│
└─────────┬───────────┬───────────────┬───────────────────────────┘
          │           │               │
┌─────────▼─────────┐ │ ┌─────────────▼─────────────────────────┐
│ api_client.py     │ │ │        embedding_client.py            │
└─────────┬─────────┘ │ └─────────────────┬─────────────────────┘
          │           │                   │
┌─────────▼─────────┐ │ ┌─────────────────▼─────────────────────┐
│  templates.py     │ │ │          templates.py                  │
└───────────────────┘ │ └─────────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                   database_clients                           │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐│
│  │graph_database   │vector_database  │       qa_generator       ││
│  │    _client.py   │   _client.py    │                          ││
│  └─────────────────┴─────────────────┴─────────────────────────┘│
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────────────┐
│                        utils/                                 │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐│
│  │   config_utils  │  file_utils     │     validation_utils    ││
│  └─────────────────┴─────────────────┴─────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 核心依赖关系

1. **concept_graph.py** 依赖:
   - `api_client.py` - API调用
   - `embedding_client.py` - 文本向量化
   - `graph_database_client.py` - 图数据库操作
   - `vector_database_client.py` - 向量数据库操作
   - `templates.py` - 提示词模板

2. **embedding_client.py** 依赖:
   - `config_utils.py` - 配置管理
   - `file_utils.py` - 文件操作

3. **database_clients** 依赖:
   - `config_utils.py` - 配置管理
   - `validation_utils.py` - 数据验证

4. **qa_generator.py** 依赖:
   - `api_client.py` - API调用
   - `concept_graph.py` - 图结构访问

## 设计模式

### 1. 工厂模式 (Factory Pattern)

```python
class DatabaseClientFactory:
    """数据库客户端工厂"""

    @staticmethod
    def create_graph_client(config: Dict) -> GraphDatabaseClient:
        db_type = config.get('type', 'neo4j')

        if db_type == 'neo4j':
            return Neo4jClient(config)
        elif db_type == 'arangodb':
            return ArangoDBClient(config)
        else:
            raise ValueError(f"Unsupported graph database: {db_type}")

# 使用示例
client = DatabaseClientFactory.create_graph_client(config)
```

### 2. 策略模式 (Strategy Pattern)

```python
class ConceptValidationStrategy(ABC):
    """概念验证策略抽象基类"""

    @abstractmethod
    def validate(self, concept: str, center_concept: str) -> float:
        pass

class SemanticSimilarityStrategy(ConceptValidationStrategy):
    """语义相似度验证策略"""

    def validate(self, concept: str, center_concept: str) -> float:
        # 实现语义相似度计算
        pass

class ConceptQualityStrategy(ConceptValidationStrategy):
    """概念质量验证策略"""

    def validate(self, concept: str, center_concept: str) -> float:
        # 实现概念质量评估
        pass
```

### 3. 观察者模式 (Observer Pattern)

```python
class ProgressObserver(ABC):
    """进度观察者"""

    @abstractmethod
    def on_progress_update(self, progress: Dict):
        pass

class ProgressTracker:
    """进度跟踪器"""

    def __init__(self):
        self.observers = []

    def add_observer(self, observer: ProgressObserver):
        self.observers.append(observer)

    def notify_progress(self, progress: Dict):
        for observer in self.observers:
            observer.on_progress_update(progress)
```

### 4. 装饰器模式 (Decorator Pattern)

```python
class CachingDecorator:
    """缓存装饰器"""

    def __init__(self, api_client):
        self.api_client = api_client
        self.cache = {}

    def chat_completion(self, messages: List[Dict], **kwargs):
        cache_key = self._generate_cache_key(messages, kwargs)

        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.api_client.chat_completion(messages, **kwargs)
        self.cache[cache_key] = result
        return result
```

### 5. 单例模式 (Singleton Pattern)

```python
class ConfigurationManager:
    """配置管理器单例"""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_config(self, config_path: str):
        if self._config is None:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)

    def get_config(self) -> Dict:
        return self._config
```

## 扩展指南

### 1. 添加新的数据库支持

要添加新的图数据库支持，需要：

1. 实现抽象基类

```python
class NewGraphDatabaseClient(GraphDatabaseClient):
    """新图数据库客户端实现"""

    def __init__(self, config: Dict):
        self.config = config
        self.driver = None

    def connect(self) -> bool:
        # 实现连接逻辑
        pass

    def add_node(self, node_id: str, labels: List[str], properties: Dict) -> bool:
        # 实现节点添加逻辑
        pass

    def add_edge(self, source: str, target: str, relationship: str, properties: Dict) -> bool:
        # 实现边添加逻辑
        pass

    def query(self, query: str, parameters: Dict = None) -> List[Dict]:
        # 实现查询逻辑
        pass
```

2. 更新工厂类

```python
def create_graph_client(config: Dict) -> GraphDatabaseClient:
    db_type = config.get('type')

    if db_type == 'neo4j':
        return Neo4jClient(config)
    elif db_type == 'new_db':
        return NewGraphDatabaseClient(config)
    # ...
```

3. 更新配置文件

```yaml
graph_database:
  type: "new_db"
  new_db:
    host: "localhost"
    port: 1234
    username: "user"
    password: "pass"
```

### 2. 添加新的向量化模型

1. 扩展EmbeddingClient

```python
class NewModelEmbeddingClient(EmbeddingClient):
    """新模型向量化客户端"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = self._load_new_model(config)

    def _load_new_model(self, config: Dict):
        # 实现新模型加载逻辑
        pass

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        # 实现新模型编码逻辑
        pass
```

2. 更新配置

```yaml
embedding:
  model_type: "new_model"
  new_model:
    model_path: "/path/to/model"
    config_file: "/path/to/config"
```

### 3. 添加新的验证策略

```python
class CustomValidationStrategy(ConceptValidationStrategy):
    """自定义验证策略"""

    def validate(self, concept: str, center_concept: str) -> float:
        # 实现自定义验证逻辑
        score = 0.0

        # 添加验证规则
        if self._check_custom_rule1(concept):
            score += 0.3

        if self._check_custom_rule2(concept, center_concept):
            score += 0.4

        return min(score, 1.0)
```

### 4. 添加新的导出格式

```python
class CustomExporter:
    """自定义导出器"""

    def export(self, graph: nx.Graph, output_path: str):
        # 实现自定义导出逻辑
        data = self._convert_graph_to_custom_format(graph)

        with open(output_path, 'w', encoding='utf-8') as f:
            self._write_custom_format(data, f)

    def _convert_graph_to_custom_format(self, graph: nx.Graph):
        # 格式转换逻辑
        pass

    def _write_custom_format(self, data: Dict, file):
        # 文件写入逻辑
        pass
```

### 5. 扩展配置系统

```python
# 添加新的配置验证器
class CustomConfigValidator:
    """自定义配置验证器"""

    def validate(self, config: Dict) -> List[str]:
        errors = []

        # 验证自定义配置项
        if 'custom_section' in config:
            custom_config = config['custom_section']
            if not self._validate_custom_config(custom_config):
                errors.append("自定义配置验证失败")

        return errors

    def _validate_custom_config(self, custom_config: Dict) -> bool:
        # 实现自定义配置验证逻辑
        pass
```

这个项目结构说明文档详细描述了MemCube Political系统的架构设计、模块组织和扩展方法。开发者可以基于这个结构快速理解系统并进行功能扩展。