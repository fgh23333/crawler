# 用户手册 - MemCube Political

## 目录
1. [系统概述](#系统概述)
2. [快速入门](#快速入门)
3. [详细使用流程](#详细使用流程)
4. [高级功能](#高级功能)
5. [结果分析](#结果分析)
6. [故障排除](#故障排除)
7. [最佳实践](#最佳实践)

## 系统概述

MemCube Political 是一个智能的政治理论概念图谱扩增系统。它能够：

- 从种子概念出发，自动构建完整的政治理论知识图谱
- 使用大语言模型进行概念推理和关系挖掘
- 支持多种图数据库和向量数据库
- 自动生成QA数据集用于问答系统训练
- 提供可视化的知识图谱展示

### 核心工作流程

```
种子概念 → 概念扩增 → 向量化存储 → 关系验证 → 迭代优化 → 知识图谱
```

## 快速入门

### 第一次运行

```bash
# 1. 确保已安装依赖
pip install -r requirements.txt

# 2. 配置API密钥
cp config/api_keys.yaml.example config/api_keys.yaml
# 编辑文件，填入你的API密钥

# 3. 运行主程序
python main.py
```

### 使用预定义种子概念

系统预置了政治理论相关的种子概念，包括：
- 马克思主义理论概念
- 西方政治思想概念
- 现代政治理论概念
- 中国政治理论概念

### 快速验证安装

```python
from src.concept_graph import ConceptExpander

# 创建扩增器实例
expander = ConceptExpander('config/config.yaml')

# 测试连接
if expander.test_connections():
    print("✅ 所有连接正常")
else:
    print("❌ 连接测试失败")
```

## 详细使用流程

### 步骤1: 环境准备

#### 1.1 检查Python环境
```bash
python --version  # 确保版本 >= 3.8
```

#### 1.2 安装依赖
```bash
pip install -r requirements.txt
```

#### 1.3 配置文件设置

**API密钥配置** (`config/api_keys.yaml`):
```yaml
gemini:
  api_key: "your_gemini_api_key_here"

openai:
  api_key: "your_openai_api_key_here"
  # 其他配置...
```

**系统配置** (`config/config.yaml`):
```yaml
api:
  model_expander: "gemini-2.5-flash"
  temperature: 0.7
  max_tokens: 32768

concept_expansion:
  max_iterations: 10
  batch_size: 50
  similarity_threshold: 0.80
```

### 步骤2: 初始化系统

```python
from src.concept_graph import ConceptExpander

# 初始化扩增器
expander = ConceptExpander('config/config.yaml')

# 检查系统状态
status = expander.get_system_status()
print(f"系统状态: {status}")
```

### 步骤3: 准备种子概念

#### 使用默认种子概念
```python
# 加载默认种子概念
expander.load_seed_concepts('data/seed_concepts.txt')
```

#### 使用自定义种子概念
```python
# 自定义种子概念列表
custom_concepts = [
    "马克思主义",
    "社会主义",
    "资本主义",
    "民主",
    "自由"
]

# 设置种子概念
expander.set_seed_concepts(custom_concepts)
```

#### 从文件加载种子概念
```python
# 种子概念文件格式 (每行一个概念)
with open('my_concepts.txt', 'r', encoding='utf-8') as f:
    concepts = [line.strip() for line in f if line.strip()]

expander.set_seed_concepts(concepts)
```

### 步骤4: 运行概念扩增

#### 运行完整扩增流程
```python
# 运行完整扩增
results = expander.run_full_expansion()

print(f"扩增完成，共进行 {len(results)} 轮迭代")
```

#### 分步运行扩增
```python
# 运行单轮迭代
iteration_result = expander.run_expansion_iteration()
print(f"本轮新增节点: {iteration_result['nodes_added']}")
print(f"本轮新增边: {iteration_result['edges_added']}")

# 检查收敛状态
convergence = expander.check_convergence()
if convergence['is_converged']:
    print(f"系统已收敛: {convergence['convergence_reason']}")
```

### 步骤5: 结果查看和导出

#### 查看扩增结果
```python
# 获取当前图统计信息
metrics = expander.calculate_metrics()
print(f"总节点数: {metrics['nodes']}")
print(f"总边数: {metrics['edges']}")
print(f"平均度数: {metrics['avg_degree']:.2f}")

# 查看概念列表
concepts = list(expander.graph.nodes())
print(f"概念数量: {len(concepts)}")
print("前10个概念:", concepts[:10])
```

#### 导出结果
```python
# 导出到JSON格式
expander.export_graph_json('results/concept_graph.json')

# 导出到GraphML格式（可导入Neo4j等图数据库）
expander.export_graph_graphml('results/concept_graph.graphml')

# 导出到CSV格式（便于分析）
expander.export_graph_csv('results/')
```

## 高级功能

### 1. 自定义概念验证

```python
from src.concept_graph import ConceptExpander

class CustomConceptExpander(ConceptExpander):
    def _validate_new_concepts(self, concepts, center_concept):
        # 调用父类验证
        base_validated = super()._validate_new_concepts(concepts, center_concept)

        # 添加自定义验证逻辑
        custom_validated = []
        for concept in base_validated:
            if self._custom_validation_rule(concept):
                custom_validated.append(concept)
            else:
                self.validity_stats["custom_filtered"] += 1

        return custom_validated

    def _custom_validation_rule(self, concept):
        # 自定义验证规则
        # 例如：只接受包含特定关键词的概念
        political_keywords = ['政治', '经济', '社会', '文化']
        return any(keyword in concept for keyword in political_keywords)

# 使用自定义验证器
expander = CustomConceptExpander('config/config.yaml')
```

### 2. 批量QA生成

```python
from src.qa_generator import QAGenerator

# 初始化QA生成器
qa_gen = QAGenerator('config/config.yaml')

# 基于概念图谱生成QA数据
qa_pairs = qa_gen.generate_qa_from_graph(
    expander.graph,
    max_qa_pairs=1000,
    difficulty_levels=['简单', '中等', '困难']
)

# 导出QA数据
qa_gen.export_qa_pairs(qa_pairs, 'results/qa_dataset.json')
```

### 3. 概念关系分析

```python
# 分析概念关系
def analyze_concept_relationships(expander):
    relationships = {}

    for concept in expander.graph.nodes():
        neighbors = list(expander.graph.neighbors(concept))
        relationships[concept] = {
            'neighbors': neighbors,
            'degree': len(neighbors),
            'clustering_coefficient': nx.clustering(expander.graph, concept)
        }

    # 找出最重要的概念（度数最高的前10个）
    top_concepts = sorted(
        relationships.items(),
        key=lambda x: x[1]['degree'],
        reverse=True
    )[:10]

    print("最重要的概念（按度数排序）:")
    for concept, data in top_concepts:
        print(f"  {concept}: 度数={data['degree']}, 聚类系数={data['clustering_coefficient']:.3f}")

    return relationships

# 执行分析
relationships = analyze_concept_relationships(expander)
```

### 4. 图谱可视化

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_concept_graph(expander, output_file='results/graph_visualization.png'):
    plt.figure(figsize=(15, 10))

    # 使用spring layout布局
    pos = nx.spring_layout(expander.graph, k=1, iterations=50)

    # 绘制节点
    node_sizes = [expander.graph.degree(node) * 50 for node in expander.graph.nodes()]
    nx.draw_networkx_nodes(expander.graph, pos, node_size=node_sizes,
                          node_color='lightblue', alpha=0.7)

    # 绘制边
    nx.draw_networkx_edges(expander.graph, pos, alpha=0.3)

    # 绘制标签（只显示重要节点的标签）
    important_nodes = [node for node, degree in expander.graph.degree() if degree > 5]
    labels = {node: node for node in important_nodes}
    nx.draw_networkx_labels(expander.graph, pos, labels, font_size=8)

    plt.title("政治理论概念图谱")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"图谱已保存到: {output_file}")

# 生成可视化
visualize_concept_graph(expander)
```

## 结果分析

### 1. 增收性分析

```python
def analyze_convergence(expander):
    convergence_history = expander.convergence_history

    print("=== 收敛性分析 ===")
    for i, info in enumerate(convergence_history):
        print(f"轮次 {i+1}:")
        print(f"  节点增长率: {info['node_growth_rate']:.4f}")
        print(f"  边增长率: {info['edge_growth_rate']:.4f}")
        if info['is_converged']:
            print(f"  收敛原因: {info['convergence_reason']}")
        print()

    # 绘制收敛曲线
    import matplotlib.pyplot as plt

    iterations = list(range(1, len(convergence_history) + 1))
    node_growth = [info['node_growth_rate'] for info in convergence_history]
    edge_growth = [info['edge_growth_rate'] for info in convergence_history]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, node_growth, 'b-', label='节点增长率')
    plt.plot(iterations, edge_growth, 'r-', label='边增长率')
    plt.xlabel('迭代次数')
    plt.ylabel('增长率')
    plt.title('概念图谱收敛曲线')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### 2. 概念质量评估

```python
def evaluate_concept_quality(expander):
    print("=== 概念质量评估 ===")

    # 统计概念有效性
    print(f"总概念数: {len(expander.concept_validity)}")
    print(f"有效概念: {expander.validity_stats['valid']}")
    print(f"无效概念: {expander.validity_stats['invalid']}")
    print(f"有效率: {expander.validity_stats['valid'] / len(expander.concept_validity) * 100:.2f}%")

    # 找出质量最高的概念
    quality_scores = expander.concept_validity
    top_quality_concepts = sorted(
        quality_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("\n质量最高的10个概念:")
    for concept, score in top_quality_concepts:
        print(f"  {concept}: {score:.3f}")
```

### 3. 知识图谱统计

```python
def comprehensive_graph_analysis(expander):
    G = expander.graph

    print("=== 知识图谱全面分析 ===")

    # 基本统计
    print(f"节点数: {G.number_of_nodes()}")
    print(f"边数: {G.number_of_edges()}")
    print(f"平均度数: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

    # 图的连通性
    print(f"连通分量数: {nx.number_connected_components(G)}")
    largest_cc = max(nx.connected_components(G), key=len)
    print(f"最大连通分量大小: {len(largest_cc)}")

    # 中心性指标
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # 找出最重要的概念
    print("\n最重要的概念（按度中心性）:")
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    for concept, centrality in top_degree:
        print(f"  {concept}: {centrality:.3f}")
```

## 故障排除

### 常见问题及解决方案

#### 1. API调用失败
```python
# 问题: 模型API调用失败
# 解决方案: 检查API密钥和网络连接

import sys
from src.api_client import APIClient

client = APIClient()
test_response = client.chat_completion(
    messages=[{"role": "user", "content": "测试"}],
    model="gemini-2.5-flash"
)

if test_response.success:
    print("✅ API连接正常")
else:
    print(f"❌ API连接失败: {test_response.error}")
    print("请检查:")
    print("1. API密钥是否正确")
    print("2. 网络连接是否正常")
    print("3. API配额是否充足")
```

#### 2. 数据库连接问题
```python
# 问题: Neo4j或Qdrant连接失败
# 解决方案: 检查服务状态和配置

def test_database_connections(expander):
    # 测试图数据库连接
    if expander.graph_client:
        try:
            result = expander.graph_client.test_connection()
            print(f"图数据库连接: {'✅ 正常' if result else '❌ 失败'}")
        except Exception as e:
            print(f"❌ 图数据库连接失败: {e}")

    # 测试向量数据库连接
    if expander.vector_search:
        try:
            stats = expander.vector_search.get_collection_stats()
            print(f"向量数据库连接: ✅ 正常 (文档数: {stats.get('count', 0)})")
        except Exception as e:
            print(f"❌ 向量数据库连接失败: {e}")
```

#### 3. 内存不足问题
```python
# 问题: 处理大量概念时内存不足
# 解决方案: 使用批处理和清理策略

class MemoryOptimizedExpander(ConceptExpander):
    def run_expansion_iteration(self):
        # 在每次迭代后清理缓存
        result = super().run_expansion_iteration()

        # 定期清理embedding缓存
        if len(self.concept_embeddings) > 10000:
            self._cleanup_embeddings()

        return result

    def _cleanup_embeddings(self):
        # 保留重要概念的embedding
        important_concepts = set()
        for node, degree in self.graph.degree():
            if degree > 3:  # 保留度数大于3的概念
                important_concepts.add(node)

        # 清理不重要的embedding
        self.concept_embeddings = {
            k: v for k, v in self.concept_embeddings.items()
            if k in important_concepts
        }

        print(f"清理后embedding缓存大小: {len(self.concept_embeddings)}")
```

#### 4. 概念扩增质量差
```python
# 问题: 扩增的概念质量不高
# 解决方案: 调整验证阈值和提示词

# 在config.yaml中调整配置
concept_expansion:
  validity_threshold: 0.7  # 提高阈值，更严格的筛选
  similarity_threshold: 0.85  # 提高相似度阈值
  max_new_concepts_per_center: 10  # 限制每个中心概念的新概念数量

api:
  temperature: 0.3  # 降低温度，减少随机性
  max_tokens: 1000  # 减少最大token数，提高精确性
```

### 日志分析

```python
import logging
from datetime import datetime

def analyze_logs(log_file='logs/concept_expansion.log'):
    """分析运行日志，发现问题"""

    error_count = 0
    warning_count = 0
    api_failures = 0

    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'ERROR' in line:
                error_count += 1
                print(f"错误: {line.strip()}")
            elif 'WARNING' in line:
                warning_count += 1
            elif 'API调用失败' in line or 'model overloaded' in line:
                api_failures += 1

    print(f"\n=== 日志统计 ===")
    print(f"错误数: {error_count}")
    print(f"警告数: {warning_count}")
    print(f"API失败数: {api_failures}")
```

## 最佳实践

### 1. 种子概念选择建议

- **选择核心概念**: 从理论的核心概念开始
- **概念覆盖性**: 确保覆盖理论的主要方面
- **概念层次性**: 包含不同抽象层次的概念
- **避免重复**: 避免概念之间的语义重叠

**示例种子概念**:
```python
political_seed_concepts = [
    # 核心理论概念
    "马克思主义", "社会主义", "资本主义",
    # 政治制度概念
    "民主", "专制", "共和",
    # 经济概念
    "市场经济", "计划经济", "分配制度",
    # 社会概念
    "阶级", "平等", "正义"
]
```

### 2. 参数调优建议

```yaml
# 推荐配置参数
concept_expansion:
  # 初期探索阶段
  max_iterations: 15
  similarity_threshold: 0.75  # 相对宽松，允许更多探索
  validity_threshold: 0.6     # 适中的质量要求

  # 精细化阶段
  # similarity_threshold: 0.85  # 更严格，减少噪声
  # validity_threshold: 0.8     # 更高质量要求

api:
  # 平衡创造性和准确性
  temperature: 0.5
  max_tokens: 2000  # 适中的响应长度

graph_database:
  # 适合中小规模图谱
  batch_size: 25
  connection_pool_size: 10
```

### 3. 数据库配置建议

#### Neo4j配置优化
```yaml
graph_database:
  neo4j:
    # 连接池配置
    max_connection_pool_size: 20
    connection_acquisition_timeout: 60

    # 批处理配置
    batch_size: 50
    batch_timeout: 60

    # 重试配置
    retry_attempts: 3
    retry_delay: 1.0
```

#### Qdrant配置优化
```yaml
vector_database:
  qdrant:
    # 向量配置
    vector_size: 1024
    distance: "Cosine"

    # 性能配置
    collection_name: "political_concepts"
    search_top_k: 20
    batch_size: 100
```

### 4. 监控和维护

```python
# 定期健康检查
def health_check(expander):
    """系统健康检查"""

    checks = {
        'api_connection': False,
        'graph_database': False,
        'vector_database': False,
        'memory_usage': False
    }

    # API连接检查
    try:
        response = expander.client.chat_completion(
            messages=[{"role": "user", "content": "健康检查"}],
            model="gemini-2.5-flash",
            max_tokens=10
        )
        checks['api_connection'] = response.success
    except:
        pass

    # 数据库连接检查
    if expander.graph_client:
        checks['graph_database'] = expander.graph_client.test_connection()

    if expander.vector_search:
        try:
            stats = expander.vector_search.get_collection_stats()
            checks['vector_database'] = True
        except:
            pass

    # 内存使用检查
    import psutil
    memory_percent = psutil.virtual_memory().percent
    checks['memory_usage'] = memory_percent < 80

    # 生成健康报告
    print("=== 系统健康检查 ===")
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}: {'正常' if status else '异常'}")

    return all(checks.values())

# 定期执行健康检查
import schedule
schedule.every(1).hours.do(lambda: health_check(expander))
```

通过遵循本手册的指导，您将能够高效地使用MemCube Political系统构建高质量的政治理论概念图谱。