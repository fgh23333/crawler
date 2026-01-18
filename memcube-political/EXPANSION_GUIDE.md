# 概念扩增配置指南

## 📋 概述

MemCube Political 提供了灵活的概念扩增配置，支持多种运行模式，从快速测试到完整的多轮扩增。

## 🚀 快速开始

### 方法1: 使用单轮扩增脚本（推荐用于测试）

```bash
python single_round_expansion.py
```

这个脚本会：
- 只执行1轮概念扩增
- 自动保存结果到Neo4j
- 提供详细的执行日志
- 避免模型过载问题

### 方法2: 使用配置管理工具

```bash
python configure_expansion.py
```

选择预设模式：
1. **测试模式** - 只扩增1轮，保存到Neo4j
2. **快速模式** - 扩增3轮，保存到Neo4j
3. **标准模式** - 扩增10轮，收敛后停止
4. **自定义模式** - 自定义所有参数

### 方法3: 手动编辑配置文件

编辑 `config/config.yaml` 中的 `concept_expansion` 部分。

## ⚙️ 配置参数详解

### 核心参数

```yaml
concept_expansion:
  max_iterations: 1                    # 最大迭代次数
  batch_size: 10                       # 批处理大小
  max_workers: 2                        # 并发工作数
  similarity_threshold: 0.80            # 概念相似度阈值
  new_concept_rate_threshold: 0.10      # 新概念增长率阈值
  new_edge_rate_threshold: 0.05         # 新边增长率阈值
```

### 运行控制参数

```yaml
concept_expansion:
  auto_save_after_iteration: true       # 每轮迭代后自动保存
  save_to_neo4j_after_each_iteration: true  # 每轮迭代后保存到Neo4j
  stop_after_first_iteration: true      # 第一轮后停止（测试用）
```

## 🎯 推荐配置

### 初次测试（推荐）

```yaml
concept_expansion:
  max_iterations: 1
  batch_size: 5
  max_workers: 1
  auto_save_after_iteration: true
  save_to_neo4j_after_each_iteration: true
  stop_after_first_iteration: true
```

### 快速验证

```yaml
concept_expansion:
  max_iterations: 3
  batch_size: 8
  max_workers: 2
  auto_save_after_iteration: true
  save_to_neo4j_after_each_iteration: true
  stop_after_first_iteration: false
```

### 生产环境

```yaml
concept_expansion:
  max_iterations: 10
  batch_size: 15
  max_workers: 3
  auto_save_after_iteration: false
  save_to_neo4j_after_each_iteration: false
  stop_after_first_iteration: false
```

## 🔧 使用场景

### 场景1: 快速验证概念质量

```bash
# 1. 配置测试模式
python configure_expansion.py
# 选择 "1. 测试模式"

# 2. 运行单轮扩增
python single_round_expansion.py

# 3. 查看结果
# - Neo4j浏览器查看概念图
# - results/目录查看统计信息
```

### 场景2: 批量处理多轮扩增

```bash
# 1. 配置标准模式
python configure_expansion.py
# 选择 "3. 标准模式"

# 2. 运行完整扩增
python main.py --stage concept-expansion

# 3. 监控进展
# - 查看日志中的收敛信息
# - 检查系统资源使用情况
```

### 场景3: 调优参数

```bash
# 1. 查看当前配置
python configure_expansion.py
# 选择 "5. 查看当前配置"

# 2. 自定义配置
python configure_expansion.py
# 选择 "4. 自定义模式"

# 3. 测试新配置
python single_round_expansion.py
```

## 📊 性能优化建议

### 避免模型过载

如果遇到模型过载：

```yaml
# 降低并发和批处理
concept_expansion:
  batch_size: 5        # 减少到5
  max_workers: 1       # 单线程

api:
  max_tokens: 2048     # 减少token数
  rate_limit_delay: 5.0 # 增加延迟

embedding:
  request_delay: 3.0  # 3秒间隔
  batch_size: 2       # 小批量
```

### 提高处理速度

如果系统资源充足：

```yaml
concept_expansion:
  batch_size: 20       # 增大批处理
  max_workers: 5       # 增加并发
```

## 🔍 结果查看

### Neo4j浏览器

访问 http://localhost:7474 查看概念图：
- 节点标签：`Concept`, `PoliticalTheory`
- 关系类型：`RELATED_TO`
- 属性：`name`, `validity_score`, `embedding`

### 统计信息

```python
# 查看Neo4j统计
MATCH (n:Concept) RETURN count(n) as node_count
MATCH ()-[r:RELATED_TO]-() RETURN count(r) as edge_count
```

### 日志分析

```bash
# 查看执行日志
tail -f logs/concept_expansion.log

# 查看数据库操作日志
tail -f logs/database_operations.log
```

## 🚨 故障排除

### 常见问题

1. **模型过载**
   - 减少批处理大小和并发数
   - 增加请求延迟
   - 参考上面的性能优化建议

2. **Neo4j连接失败**
   - 检查Neo4j是否在运行
   - 验证配置文件中的连接参数
   - 运行 `python scripts/test_connections.py`

3. **内存不足**
   - 减少批处理大小
   - 关闭其他应用程序
   - 监控系统资源使用

### 调试模式

```bash
# 检查环境
python main.py --check-env

# 测试API
python main.py --test-api

# 测试系统
python main.py --test-system
```

## 📈 扩展配置

### 运行不同规模的扩增

```yaml
# 小规模测试 (100-500个概念)
concept_expansion:
  max_iterations: 2
  batch_size: 5

# 中等规模 (500-2000个概念)
concept_expansion:
  max_iterations: 5
  batch_size: 10

# 大规模 (2000+个概念)
concept_expansion:
  max_iterations: 10
  batch_size: 20
```

### 调整收敛阈值

```yaml
concept_expansion:
  # 更严格的收敛条件
  similarity_threshold: 0.85        # 提高相似度阈值
  new_concept_rate_threshold: 0.05  # 降低新概念增长率

  # 更宽松的收敛条件
  similarity_threshold: 0.70        # 降低相似度阈值
  new_concept_rate_threshold: 0.20  # 提高新概念增长率
```

## 🎉 下一步

完成概念扩增后，你可以：

1. **生成QA数据** - 基于概念图谱生成问答对
2. **构建知识库** - 将概念图转换为可查询的知识库
3. **性能评估** - 评估概念图谱的质量和覆盖率
4. **可视化分析** - 使用工具可视化概念关系

更多信息请参考：
- [用户手册](docs/USER_MANUAL.md)
- [API参考](docs/API_REFERENCE.md)
- [项目结构](docs/PROJECT_STRUCTURE.md)