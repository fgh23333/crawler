# API参考文档 - MemCube Political

## 目录
1. [核心API概览](#核心api概览)
2. [概念扩增API](#概念扩增api)
3. [向量化API](#向量化api)
4. [图数据库API](#图数据库api)
5. [向量数据库API](#向量数据库api)
6. [QA生成API](#qa生成api)
7. [数据结构](#数据结构)
8. [错误处理](#错误处理)
9. [示例代码](#示例代码)

## 核心API概览

MemCube Political 提供了一套完整的API用于政治理论概念图谱的构建、扩增和查询。核心组件包括：

- **ConceptExpander** - 概念扩增引擎
- **EmbeddingClient** - 文本向量化客户端
- **GraphDatabaseClient** - 图数据库客户端
- **VectorDatabaseClient** - 向量数据库客户端
- **QAGenerator** - QA数据生成器

### 导入和初始化

```python
# 基础导入
from src.concept_graph import ConceptExpander
from src.embedding_client import EmbeddingClient
from src.qa_generator import QAGenerator
from src.graph_database_client import get_graph_client
from src.vector_database_client import get_vector_client

# 初始化主要组件
expander = ConceptExpander('config/config.yaml')
embedding_client = EmbeddingClient('config/config.yaml')
qa_generator = QAGenerator('config/config.yaml')
```

## 概念扩增API

### ConceptExpander 类

概念扩增的核心引擎，负责从种子概念出发构建完整的知识图谱。

#### 构造函数

```python
def __init__(self, config_path: str)
```

**参数:**
- `config_path` (str): 配置文件路径

**返回:**
- `ConceptExpander`: 概念扩增器实例

**示例:**
```python
expander = ConceptExpander('config/config.yaml')
```

#### 核心方法

##### load_seed_concepts()

```python
def load_seed_concepts(self, concepts_file: str) -> None
```

**功能:** 从文件加载种子概念

**参数:**
- `concepts_file` (str): 概念文件路径，每行一个概念

**示例:**
```python
expander.load_seed_concepts('data/political_concepts.txt')
```

##### set_seed_concepts()

```python
def set_seed_concepts(self, concepts: List[str]) -> None
```

**功能:** 直接设置种子概念列表

**参数:**
- `concepts` (List[str]): 种子概念列表

**示例:**
```python
concepts = ["马克思主义", "社会主义", "民主"]
expander.set_seed_concepts(concepts)
```

##### run_full_expansion()

```python
def run_full_expansion(self) -> List[Dict[str, Any]]
```

**功能:** 运行完整的概念扩增流程

**返回:**
- `List[Dict]`: 迭代结果列表，每个元素包含：
  - `iteration` (int): 迭代次数
  - `metrics` (Dict): 图统计指标
  - `nodes_added` (int): 新增节点数
  - `edges_added` (int): 新增边数
  - `timestamp` (str): 时间戳

**示例:**
```python
results = expander.run_full_expansion()
print(f"扩增完成，共进行 {len(results)} 轮迭代")
```

##### run_expansion_iteration()

```python
def run_expansion_iteration(self) -> Dict[str, Any]
```

**功能:** 运行单轮概念扩增迭代

**返回:**
- `Dict`: 单轮迭代结果，包含：
  - `iteration` (int): 迭代次数
  - `metrics` (Dict): 图统计指标
  - `batch_results` (List): 批处理结果
  - `nodes_added` (int): 新增节点数
  - `edges_added` (int): 新增边数

**示例:**
```python
iteration_result = expander.run_expansion_iteration()
print(f"本轮新增 {iteration_result['nodes_added']} 个概念")
```

##### expand_single_concept()

```python
def expand_single_concept(
    self,
    center_concept: str,
    neighbors: List[str],
    concept_id: str
) -> ConceptExpansionResult
```

**功能:** 扩增单个概念

**参数:**
- `center_concept` (str): 中心概念
- `neighbors` (List[str]): 邻居概念列表
- `concept_id` (str): 概念ID

**返回:**
- `ConceptExpansionResult`: 扩增结果对象

**示例:**
```python
result = expander.expand_single_concept(
    center_concept="马克思主义",
    neighbors=["社会主义", "资本主义"],
    concept_id="concept_001"
)

if result.status == "success":
    print(f"新增概念: {result.new_concepts}")
```

##### check_convergence()

```python
def check_convergence(
    self,
    previous_metrics: Optional[Dict] = None
) -> Dict[str, Any]
```

**功能:** 检查系统是否收敛

**参数:**
- `previous_metrics` (Dict, optional): 上一轮的指标

**返回:**
- `Dict`: 收敛信息，包含：
  - `is_converged` (bool): 是否收敛
  - `convergence_reason` (str): 收敛原因
  - `node_growth_rate` (float): 节点增长率
  - `edge_growth_rate` (float): 边增长率

**示例:**
```python
convergence = expander.check_convergence()
if convergence['is_converged']:
    print(f"系统已收敛: {convergence['convergence_reason']}")
```

##### calculate_metrics()

```python
def calculate_metrics(self) -> Dict[str, float]
```

**功能:** 计算当前图的统计指标

**返回:**
- `Dict`: 统计指标，包含：
  - `nodes` (int): 节点数
  - `edges` (int): 边数
  - `avg_degree` (float): 平均度数
  - `density` (float): 图密度
  - `clustering_coefficient` (float): 聚类系数

**示例:**
```python
metrics = expander.calculate_metrics()
print(f"当前图有 {metrics['nodes']} 个节点，{metrics['edges']} 条边")
```

#### 导出方法

##### export_graph_json()

```python
def export_graph_json(self, output_file: str) -> None
```

**功能:** 导出图到JSON文件

**参数:**
- `output_file` (str): 输出文件路径

**示例:**
```python
expander.export_graph_json('results/concept_graph.json')
```

##### export_graph_graphml()

```python
def export_graph_graphml(self, output_file: str) -> None
```

**功能:** 导出图到GraphML格式

**参数:**
- `output_file` (str): 输出文件路径

**示例:**
```python
expander.export_graph_graphml('results/concept_graph.graphml')
```

##### export_graph_csv()

```python
def export_graph_csv(self, output_dir: str) -> None
```

**功能:** 导出图到CSV格式

**参数:**
- `output_dir` (str): 输出目录路径

**示例:**
```python
expander.export_graph_csv('results/csv/')
```

## 向量化API

### EmbeddingClient 类

负责将文本转换为向量表示，支持多种模型和部署方式。

#### 构造函数

```python
def __init__(self, config: Dict[str, Any])
```

**参数:**
- `config` (Dict): 配置字典

**示例:**
```python
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

embedding_client = EmbeddingClient(config)
```

#### 核心方法

##### encode()

```python
def encode(
    self,
    texts: Union[str, List[str]],
    batch_size: Optional[int] = None
) -> Union[np.ndarray, List[np.ndarray]]
```

**功能:** 将文本编码为向量

**参数:**
- `texts` (Union[str, List[str]]): 文本或文本列表
- `batch_size` (int, optional): 批处理大小

**返回:**
- `Union[np.ndarray, List[np.ndarray]]`: 向量或向量列表

**示例:**
```python
# 单个文本
vector = embedding_client.encode("马克思主义理论")
print(f"向量维度: {vector.shape}")

# 批量处理
texts = ["马克思主义", "社会主义", "资本主义"]
vectors = embedding_client.encode(texts)
print(f"批量处理 {len(vectors)} 个文本")
```

##### encode_with_cache()

```python
def encode_with_cache(
    self,
    texts: List[str],
    cache_key: Optional[str] = None
) -> np.ndarray
```

**功能:** 带缓存的文本编码

**参数:**
- `texts` (List[str]): 文本列表
- `cache_key` (str, optional): 缓存键

**返回:**
- `np.ndarray`: 向量数组

**示例:**
```python
vectors = embedding_client.encode_with_cache(
    texts=["政治理论", "经济制度"],
    cache_key="political_concepts"
)
```

##### similarity()

```python
def similarity(
    self,
    text1: Union[str, np.ndarray],
    text2: Union[str, np.ndarray]
) -> float
```

**功能:** 计算两个文本/向量之间的相似度

**参数:**
- `text1` (Union[str, np.ndarray]): 文本1或向量1
- `text2` (Union[str, np.ndarray]): 文本2或向量2

**返回:**
- `float`: 相似度分数 (0-1)

**示例:**
```python
# 文本相似度
sim = embedding_client.similarity("马克思主义", "社会主义")
print(f"相似度: {sim:.3f}")

# 向量相似度
vec1 = embedding_client.encode("民主")
vec2 = embedding_client.encode("自由")
sim = embedding_client.similarity(vec1, vec2)
```

##### most_similar()

```python
def most_similar(
    self,
    query: Union[str, np.ndarray],
    candidates: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]
```

**功能:** 找出与查询最相似的候选项

**参数:**
- `query` (Union[str, np.ndarray]): 查询文本或向量
- `candidates` (List[str]): 候选文本列表
- `top_k` (int): 返回前k个最相似项

**返回:**
- `List[Tuple[str, float]]`: (文本, 相似度) 元组列表

**示例:**
```python
candidates = ["民主制度", "专政制度", "共和制", "君主制"]
results = embedding_client.most_similar("民主", candidates, top_k=3)

for text, score in results:
    print(f"{text}: {score:.3f}")
```

## 图数据库API

### GraphDatabaseClient 类

图数据库的统一接口，支持Neo4j、ArangoDB等。

#### 工厂函数

##### get_graph_client()

```python
def get_graph_client(
    config_path: str,
    db_type: Optional[str] = None
) -> GraphDatabaseClient
```

**功能:** 获取图数据库客户端实例

**参数:**
- `config_path` (str): 配置文件路径
- `db_type` (str, optional): 数据库类型，如不指定则从配置读取

**返回:**
- `GraphDatabaseClient`: 图数据库客户端实例

**示例:**
```python
# 从配置文件获取
client = get_graph_client('config/config.yaml')

# 指定数据库类型
neo4j_client = get_graph_client('config/config.yaml', 'neo4j')
```

#### 核心接口方法

##### connect()

```python
def connect(self) -> bool
```

**功能:** 连接到图数据库

**返回:**
- `bool`: 连接是否成功

**示例:**
```python
if client.connect():
    print("图数据库连接成功")
else:
    print("图数据库连接失败")
```

##### add_node()

```python
def add_node(
    self,
    node_id: str,
    labels: List[str],
    properties: Dict[str, Any]
) -> bool
```

**功能:** 添加节点

**参数:**
- `node_id` (str): 节点ID
- `labels` (List[str]): 节点标签列表
- `properties` (Dict): 节点属性

**返回:**
- `bool`: 是否成功

**示例:**
```python
success = client.add_node(
    node_id="concept_001",
    labels=["Concept", "PoliticalTheory"],
    properties={
        "name": "马克思主义",
        "category": "政治理论",
        "confidence": 0.95
    }
)
```

##### add_edge()

```python
def add_edge(
    self,
    source_id: str,
    target_id: str,
    relationship_type: str,
    properties: Dict[str, Any]
) -> bool
```

**功能:** 添加边

**参数:**
- `source_id` (str): 源节点ID
- `target_id` (str): 目标节点ID
- `relationship_type` (str): 关系类型
- `properties` (Dict): 边属性

**返回:**
- `bool`: 是否成功

**示例:**
```python
success = client.add_edge(
    source_id="concept_001",
    target_id="concept_002",
    relationship_type="RELATED_TO",
    properties={
        "strength": 0.8,
        "relationship_type": "理论发展"
    }
)
```

##### query()

```python
def query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]
```

**功能:** 执行图查询

**参数:**
- `cypher_query` (str): Cypher查询语句
- `parameters` (Dict, optional): 查询参数

**返回:**
- `List[Dict]`: 查询结果

**示例:**
```python
# 查找所有政治理论概念
results = client.query("""
    MATCH (c:Concept)-[:RELATED_TO]-(related)
    WHERE c.category = '政治理论'
    RETURN c.name, related.name, count(*) as connection_count
    ORDER BY connection_count DESC
    LIMIT 10
""")

for record in results:
    print(f"{record['c.name']} - {record['related.name']}: {record['connection_count']}")
```

##### batch_operations()

```python
def batch_operations(
    self,
    operations: List[Dict],
    batch_size: int = 50
) -> Dict[str, int]
```

**功能:** 批量执行数据库操作

**参数:**
- `operations` (List[Dict]): 操作列表
- `batch_size` (int): 批处理大小

**返回:**
- `Dict[str, int]`: 操作结果统计

**示例:**
```python
operations = [
    {
        "type": "node",
        "action": "create",
        "id": "concept_001",
        "labels": ["Concept"],
        "properties": {"name": "新概念"}
    },
    {
        "type": "edge",
        "action": "create",
        "source": "concept_001",
        "target": "concept_002",
        "relationship": "RELATED_TO"
    }
]

results = client.batch_operations(operations)
print(f"成功: {results['success']}, 失败: {results['failed']}")
```

## 向量数据库API

### VectorDatabaseClient 类

向量数据库的统一接口，支持Qdrant、ChromaDB、FAISS等。

#### 工厂函数

##### get_vector_client()

```python
def get_vector_client(config_path: str) -> VectorDatabaseClient
```

**功能:** 获取向量数据库客户端实例

**参数:**
- `config_path` (str): 配置文件路径

**返回:**
- `VectorDatabaseClient`: 向量数据库客户端实例

**示例:**
```python
client = get_vector_client('config/config.yaml')
```

#### 核心接口方法

##### connect()

```python
def connect(self) -> bool
```

**功能:** 连接到向量数据库

**返回:**
- `bool`: 连接是否成功

##### create_collection()

```python
def create_collection(
    self,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine"
) -> bool
```

**功能:** 创建向量集合

**参数:**
- `collection_name` (str): 集合名称
- `vector_size` (int): 向量维度
- `distance` (str): 距离计算方式

**返回:**
- `bool`: 是否成功

**示例:**
```python
success = client.create_collection(
    collection_name="political_concepts",
    vector_size=1024,
    distance="Cosine"
)
```

##### index_concepts()

```python
def index_concepts(
    self,
    collection_name: str,
    concepts: List[Dict[str, Any]]
) -> bool
```

**功能:** 索引概念向量

**参数:**
- `collection_name` (str): 集合名称
- `concepts` (List[Dict]): 概念列表，每个概念包含:
  - `id` (str): 概念ID
  - `name` (str): 概念名称
  - `vector` (List[float]): 向量
  - `metadata` (Dict): 元数据

**返回:**
- `bool`: 是否成功

**示例:**
```python
concepts = [
    {
        "id": "concept_001",
        "name": "马克思主义",
        "vector": [0.1, 0.2, 0.3, ...],
        "metadata": {"category": "政治理论", "confidence": 0.95}
    }
]

success = client.index_concepts("political_concepts", concepts)
```

##### search()

```python
def search(
    self,
    collection_name: str,
    query_vector: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.7
) -> List[Dict[str, Any]]
```

**功能:** 向量相似度搜索

**参数:**
- `collection_name` (str): 集合名称
- `query_vector` (np.ndarray): 查询向量
- `top_k` (int): 返回结果数量
- `threshold` (float): 相似度阈值

**返回:**
- `List[Dict]: 搜索结果，每个结果包含:
  - `id` (str): 概念ID
  - `name` (str): 概念名称
  - `score` (float): 相似度分数
  - `metadata` (Dict): 元数据

**示例:**
```python
import numpy as np
query_vector = embedding_client.encode("社会主义理论")
results = client.search(
    collection_name="political_concepts",
    query_vector=query_vector,
    top_k=5,
    threshold=0.7
)

for result in results:
    print(f"{result['name']}: {result['score']:.3f}")
```

##### check_concepts_exist()

```python
def check_concepts_exist(
    self,
    collection_name: str,
    concept_ids: List[str]
) -> Dict[str, bool]
```

**功能:** 检查概念是否已存在

**参数:**
- `collection_name` (str): 集合名称
- `concept_ids` (List[str]): 概念ID列表

**返回:**
- `Dict[str, bool]`: 存在性检查结果

**示例:**
```python
concept_ids = ["concept_001", "concept_002", "concept_003"]
existence = client.check_concepts_exist("political_concepts", concept_ids)

for concept_id, exists in existence.items():
    print(f"{concept_id}: {'存在' if exists else '不存在'}")
```

## QA生成API

### QAGenerator 类

基于概念图谱生成问答数据集。

#### 构造函数

```python
def __init__(self, config_path: str)
```

**参数:**
- `config_path` (str): 配置文件路径

**示例:**
```python
qa_generator = QAGenerator('config/config.yaml')
```

#### 核心方法

##### generate_qa_from_graph()

```python
def generate_qa_from_graph(
    self,
    graph: nx.Graph,
    max_qa_pairs: int = 1000,
    difficulty_levels: List[str] = None
) -> List[Dict[str, Any]]
```

**功能:** 基于图结构生成QA数据

**参数:**
- `graph` (nx.Graph): NetworkX图对象
- `max_qa_pairs` (int): 最大QA对数量
- `difficulty_levels` (List[str], optional): 难度级别列表

**返回:**
- `List[Dict]: QA对列表

**示例:**
```python
qa_pairs = qa_generator.generate_qa_from_graph(
    graph=expander.graph,
    max_qa_pairs=500,
    difficulty_levels=['简单', '中等', '困难']
)

print(f"生成 {len(qa_pairs)} 个QA对")
```

##### generate_concept_based_qa()

```python
def generate_concept_based_qa(
    self,
    concepts: List[str],
    relationships: List[Tuple[str, str, str]],
    num_questions_per_concept: int = 3
) -> List[Dict[str, Any]]
```

**功能:** 基于概念和关系生成QA

**参数:**
- `concepts` (List[str]): 概念列表
- `relationships` (List[Tuple]): 关系列表 (源, 目标, 关系类型)
- `num_questions_per_concept` (int): 每个概念的题目数量

**返回:**
- `List[Dict]: QA对列表

**示例:**
```python
concepts = ["马克思主义", "社会主义", "资本主义"]
relationships = [
    ("马克思主义", "社会主义", "发展为"),
    ("社会主义", "共产主义", "初级阶段")
]

qa_pairs = qa_generator.generate_concept_based_qa(
    concepts=concepts,
    relationships=relationships,
    num_questions_per_concept=5
)
```

##### export_qa_pairs()

```python
def export_qa_pairs(
    self,
    qa_pairs: List[Dict[str, Any]],
    output_file: str,
    format: str = "json"
) -> None
```

**功能:** 导出QA数据到文件

**参数:**
- `qa_pairs` (List[Dict]): QA对列表
- `output_file` (str): 输出文件路径
- `format` (str): 输出格式 (json, csv, jsonl)

**示例:**
```python
qa_generator.export_qa_pairs(
    qa_pairs=qa_pairs,
    output_file='results/political_theory_qa.json',
    format='json'
)
```

## 数据结构

### ConceptExpansionResult

概念扩增结果数据结构。

```python
@dataclass
class ConceptExpansionResult:
    concept_id: str           # 概念ID
    center_concept: str       # 中心概念
    status: str              # 状态: success, error, no_concepts
    new_concepts: List[str]   # 新概念列表
    returned_center: str     # 返回的中心概念
    error_message: str = ""   # 错误信息
    timestamp: str = ""      # 时间戳
    metadata: Dict = None    # 元数据
```

### APIResponse

API响应数据结构。

```python
@dataclass
class APIResponse:
    success: bool            # 是否成功
    content: Any             # 响应内容
    error: str = ""          # 错误信息
    usage: Dict = None       # 使用情况
    model: str = ""          # 模型名称
    response_time: float = 0.0  # 响应时间
```

### GraphMetrics

图统计指标数据结构。

```python
@dataclass
class GraphMetrics:
    nodes: int               # 节点数
    edges: int               # 边数
    avg_degree: float        # 平均度数
    density: float           # 图密度
    clustering_coefficient: float  # 聚类系数
    components: int          # 连通分量数
    largest_component_size: int  # 最大连通分量大小
```

### QAData

QA数据结构。

```python
@dataclass
class QAData:
    question: str            # 问题
    answer: str              # 答案
    concepts: List[str]      # 相关概念
    difficulty: str          # 难度级别
    category: str            # 分类
    source: str = ""         # 来源
    metadata: Dict = None    # 元数据
```

## 错误处理

### 异常类型

#### ConfigurationError

配置错误异常。

```python
class ConfigurationError(Exception):
    """配置相关错误"""
    pass
```

#### APIError

API调用错误异常。

```python
class APIError(Exception):
    """API调用错误"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(message)
```

#### DatabaseError

数据库操作错误异常。

```python
class DatabaseError(Exception):
    """数据库操作错误"""
    def __init__(self, message: str, operation: str = None):
        self.message = message
        self.operation = operation
        super().__init__(message)
```

### 错误处理示例

```python
try:
    expander = ConceptExpander('config/config.yaml')
    results = expander.run_full_expansion()
except ConfigurationError as e:
    print(f"配置错误: {e}")
except APIError as e:
    print(f"API调用失败: {e}")
except DatabaseError as e:
    print(f"数据库操作失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 示例代码

### 完整的概念扩增流程

```python
#!/usr/bin/env python3
"""
完整的概念扩增示例
"""

import yaml
from src.concept_graph import ConceptExpander
from src.qa_generator import QAGenerator

def main():
    # 1. 初始化配置
    config_path = 'config/config.yaml'

    # 2. 创建概念扩增器
    expander = ConceptExpander(config_path)

    # 3. 设置种子概念
    seed_concepts = [
        "马克思主义", "社会主义", "资本主义",
        "民主", "自由", "平等"
    ]
    expander.set_seed_concepts(seed_concepts)

    # 4. 测试连接
    if not expander.test_connections():
        print("数据库连接测试失败")
        return

    # 5. 运行概念扩增
    print("开始概念扩增...")
    results = expander.run_full_expansion()

    # 6. 显示结果
    final_metrics = results[-1]['metrics']
    print(f"扩增完成:")
    print(f"  总概念数: {final_metrics['nodes']}")
    print(f"  总关系数: {final_metrics['edges']}")
    print(f"  迭代次数: {len(results)}")

    # 7. 导出结果
    expander.export_graph_json('results/concept_graph.json')
    expander.export_graph_csv('results/csv/')

    # 8. 生成QA数据
    print("生成QA数据...")
    qa_generator = QAGenerator(config_path)
    qa_pairs = qa_generator.generate_qa_from_graph(
        graph=expander.graph,
        max_qa_pairs=1000
    )

    # 9. 导出QA数据
    qa_generator.export_qa_pairs(qa_pairs, 'results/qa_dataset.json')
    print(f"生成 {len(qa_pairs)} 个QA对")

    print("流程完成！")

if __name__ == "__main__":
    main()
```

### 自定义概念验证

```python
#!/usr/bin/env python3
"""
自定义概念验证示例
"""

from src.concept_graph import ConceptExpander

class CustomConceptExpander(ConceptExpander):
    def _validate_new_concepts(self, concepts, center_concept):
        """自定义概念验证逻辑"""
        # 调用父类验证
        base_validated = super()._validate_new_concepts(concepts, center_concept)

        # 添加自定义验证规则
        custom_validated = []
        for concept in base_validated:
            if self._custom_political_relevance_check(concept):
                custom_validated.append(concept)
            else:
                self.validity_stats["custom_filtered"] += 1

        return custom_validated

    def _custom_political_relevance_check(self, concept):
        """自定义政治理论相关性检查"""
        # 定义政治理论相关关键词
        political_keywords = [
            '政治', '经济', '社会', '文化', '理论', '思想',
            '制度', '民主', '自由', '平等', '权利', '权力'
        ]

        # 检查概念是否包含相关关键词
        return any(keyword in concept for keyword in political_keywords)

# 使用自定义扩增器
expander = CustomConceptExpander('config/config.yaml')
expander.set_seed_concepts(["政治理论"])
results = expander.run_full_expansion()
```

### 批量向量化示例

```python
#!/usr/bin/env python3
"""
批量向量化示例
"""

from src.embedding_client import EmbeddingClient
import numpy as np

def main():
    # 初始化向量化客户端
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    embedding_client = EmbeddingClient(config)

    # 准备文本数据
    texts = [
        "马克思主义理论体系",
        "社会主义核心价值",
        "资本主义市场经济",
        "民主政治制度",
        "自由市场经济",
        "社会主义初级阶段"
    ]

    # 批量向量化
    print("开始批量向量化...")
    vectors = embedding_client.encode(texts, batch_size=3)

    print(f"完成 {len(vectors)} 个文本的向量化")
    print(f"向量维度: {vectors[0].shape}")

    # 计算相似度矩阵
    similarity_matrix = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(len(texts)):
            if i != j:
                sim = embedding_client.similarity(vectors[i], vectors[j])
                similarity_matrix[i][j] = sim

    # 显示相似度最高的文本对
    print("\n相似度最高的文本对:")
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            sim = similarity_matrix[i][j]
            if sim > 0.7:  # 相似度阈值
                print(f"  {texts[i]} <-> {texts[j]}: {sim:.3f}")

if __name__ == "__main__":
    main()
```

这份API参考文档提供了MemCube Political系统的完整API接口说明，包括详细的方法参数、返回值和使用示例。开发者可以根据这些文档快速集成和使用系统的各项功能。