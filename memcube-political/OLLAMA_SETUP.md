# Ollama + BGE-M3 模型设置指南

## 🎯 概述

本项目配置使用本地Ollama服务运行BGE-M3 embedding模型，提供高效的中文文本向量化能力。

## 📋 环境要求

- Ollama (0.1.0+)
- bge-m3 模型
- 至少 4GB RAM
- Docker 或本地安装

## 🚀 快速设置

### 1. 安装Ollama

#### Windows
```bash
# 下载并安装Ollama
# 访问 https://ollama.com/download 下载Windows版本
# 或使用winget
winget install Ollama.Ollama
```

#### macOS
```bash
# 使用Homebrew
brew install ollama

# 或下载DMG文件
# https://ollama.com/download
```

#### Linux
```bash
# 官方安装脚本
curl -fsSL https://ollama.com/install.sh | sh

# 或使用Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### 2. 启动Ollama服务

```bash
# Windows/macOS - 从应用菜单启动
# 或命令行启动
ollama serve

# Docker
docker start ollama
```

### 3. 下载BGE-M3模型

```bash
# 下载BGE-M3模型（中文支持）
ollama pull bge-m3

# 验证模型安装
ollama list
```

### 4. 测试模型

```bash
# 测试embedding功能
curl http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "prompt": "这是一个测试文本"
  }'
```

## ⚙️ 配置验证

### 1. 检查Ollama服务状态

```bash
# 检查服务是否运行
curl http://localhost:11434/api/tags

# 应该返回类似以下内容：
# {"models":[{"name":"bge-m3:latest","modified_at":"...","size":...}]}
```

### 2. 验证模型可用性

```bash
# 列出已安装的模型
ollama list

# 确保bge-m3在列表中
# NAME            ID              SIZE    MODIFIED
# bge-m3:latest   abc123...       670MB   2025-11-10
```

### 3. 测试embedding功能

```python
# 使用Python测试
import requests

response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "bge-m3",
        "prompt": "马克思主义是中国特色社会主义的理论基础"
    }
)

if response.status_code == 200:
    embedding = response.json()['embedding']
    print(f"✅ Embedding成功，维度: {len(embedding)}")
    print(f"前5个维度: {embedding[:5]}")
else:
    print(f"❌ 错误: {response.status_code}")
```

## 🔧 配置文件

项目的配置文件已自动设置为使用Ollama：

```yaml
# config/config.yaml
embedding:
  model_name: "bge-m3"      # Ollama模型名
  model_type: "ollama"       # 使用Ollama后端
  ollama_url: "http://localhost:11434"  # Ollama服务地址
  batch_size: 16
  device: "cpu"
```

## 🐛 故障排除

### 常见问题

#### 1. Ollama连接失败
```
错误: 无法连接到Ollama服务
解决:
- 确保Ollama服务正在运行
- 检查端口11434是否被占用
- 验证防火墙设置
```

#### 2. 模型未找到
```
错误: 模型 bge-m3 未在ollama中找到
解决:
- 运行 `ollama pull bge-m3` 下载模型
- 使用 `ollama list` 确认模型安装
```

#### 3. 内存不足
```
错误: 内存不足
解决:
- 减少batch_size配置
- 关闭其他占用内存的程序
- 考虑使用更大的内存
```

#### 4. 响应慢
```
问题: embedding生成速度慢
解决:
- 确保有足够的CPU/GPU资源
- 减少batch_size
- 检查系统负载
```

### 日志检查

```bash
# 查看Ollama日志
# Windows: %USERPROFILE%\.ollama\logs
# macOS: ~/.ollama/logs
# Linux: ~/.ollama/logs

# 或者查看Docker日志
docker logs ollama
```

### 网络问题

```bash
# 测试连接
curl -I http://localhost:11434/api/tags

# 检查端口占用
netstat -an | grep 11434  # Linux/macOS
netstat -an | findstr 11434  # Windows
```

## 📊 性能优化

### 1. 批处理优化

```yaml
# config/config.yaml
embedding:
  batch_size: 32  # 根据内存调整
```

### 2. 并发控制

```yaml
# config/config.yaml
concept_expansion:
  max_workers: 5   # 减少并发避免Ollama过载
```

### 3. 缓存策略

项目会自动缓存embedding结果，减少重复计算。

## 🔄 模型管理

### 更新模型
```bash
# 更新到最新版本
ollama pull bge-m3:latest

# 或者指定版本
ollama pull bge-m3:v1.0
```

### 删除模型
```bash
# 删除不需要的模型
ollama rm bge-m3
```

### 查看模型信息
```bash
# 查看模型详细信息
ollama show bge-m3
```

## 🎛️ 高级配置

### 1. 自定义Ollama端口

```yaml
# config/config.yaml
embedding:
  ollama_url: "http://localhost:11435"  # 自定义端口
```

### 2. Docker配置

```yaml
# docker-compose.yml
version: '3'
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama:/root/.ollama
    environment:
      - OLLAMA_MAX_LOADED_MODELS=1
      - OLLAMA_NUM_PARALLEL=2
    restart: unless-stopped

volumes:
  ollama:
```

### 3. 环境变量

```bash
# 设置环境变量
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=2

# 启动服务
ollama serve
```

## 📚 相关资源

- [Ollama官方文档](https://github.com/ollama/ollama)
- [BGE-M3模型介绍](https://huggingface.co/BAAI/bge-m3)
- [项目配置文件](config/config.yaml)

## 🆘 获取帮助

如果遇到问题：

1. 查看 [Ollama GitHub Issues](https://github.com/ollama/ollama/issues)
2. 检查 [BGE-M3文档](https://huggingface.co/BAAI/bge-m3)
3. 查看项目日志文件
4. 提交Issue到项目仓库

---

*配置完成后，您可以运行 `python quick_start.py` 验证环境设置*