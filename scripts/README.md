# 脚本文件组织结构

这个目录包含了用于多线程数据请求、合并去重、按学科和题型归类并存储数据的脚本文件。

## 目录结构

### 📁 data-fetch/ (数据获取脚本)
- **main.js** - 多线程数据请求主控制器，使用Worker线程管理并发请求
- **new_index.js** - 单个Worker线程的数据请求脚本，支持9个学科的数据获取
- **multiple.js** - 增强版多线程数据请求脚本，包含错误重试机制和更好的并发控制

### 📁 data-process/ (数据处理脚本)
- **clear.js** - 数据去重脚本，移除重复的题目数据
- **merge.js** - 数据合并脚本，合并新旧数据并去重
- **classify.js** - 题目分类脚本，根据题目特征将题目分为不同类型（单选、多选、判断、填空）

### 📁 data-transform/ (数据转换脚本)
- **data_transformer.js** - 数据格式转换脚本，将分类后的数据转换为目标格式
- **concept_extractor.js** - 概念提取脚本，从题目数据中提取政治理论核心概念

### 📁 utils/ (工具脚本)
- **test.js** - 测试脚本，用于简单的功能测试

## 数据流程

1. **数据获取阶段** (data-fetch/)
   - 使用多线程并发请求数据接口
   - 支持9个学科：XiIntro, Marx, MaoIntro, Political, CMH, NCH, SDH, ORH, CCPH
   - 每个学科请求1000次，获取大量题目数据

2. **数据处理阶段** (data-process/)
   - 去除重复数据
   - 合并新旧数据
   - 按题型分类：单选、多选、判断、填空

3. **数据转换阶段** (data-transform/)
   - 转换数据格式以适应不同用途
   - 提取政治理论核心概念

## 使用方法

### 数据获取
```bash
# 运行基础多线程请求
node scripts/data-fetch/main.js

# 运行增强版多线程请求（推荐）
node scripts/data-fetch/multiple.js
```

### 数据处理
```bash
# 数据去重
node scripts/data-process/clear.js

# 数据合并
node scripts/data-process/merge.js

# 题目分类
node scripts/data-process/classify.js
```

### 数据转换
```bash
# 格式转换
node scripts/data-transform/data_transformer.js

# 概念提取
node scripts/data-transform/concept_extractor.js
```

## 注意事项

- 脚本中包含硬编码的API地址和参数
- 部分脚本依赖特定的目录结构
- 建议按顺序执行：获取 → 处理 → 转换
- 多线程脚本会创建大量并发连接，注意网络负载

## 学科映射

| 代码 | 全称 |
|------|------|
| XiIntro | 习近平新时代中国特色社会主义思想概论 |
| Marx | 马克思主义基本原理 |
| MaoIntro | 毛泽东思想和中国特色社会主义理论体系概论 |
| Political | 思想道德与法治 |
| CMH | 中国近现代史纲要 |
| NCH | 新中国史 |
| SDH | 社会主义发展史 |
| ORH | 改革开放史 |
| CCPH | 中共党史 |