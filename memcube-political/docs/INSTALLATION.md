# 安装指南 - MemCube Political

## 目录
1. [系统要求](#系统要求)
2. [Python环境安装](#python环境安装)
3. [项目安装](#项目安装)
4. [数据库安装](#数据库安装)
5. [配置文件设置](#配置文件设置)
6. [验证安装](#验证安装)
7. [故障排除](#故障排除)
8. [可选组件](#可选组件)

## 系统要求

### 最低配置要求
- **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+ / CentOS 7+
- **Python**: 3.8 或更高版本
- **内存**: 8GB RAM (推荐16GB+)
- **存储空间**: 至少10GB可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **操作系统**: Ubuntu 20.04+ / Windows 11 / macOS 12+
- **Python**: 3.9 或 3.10
- **内存**: 16GB+ RAM
- **存储空间**: 50GB+ SSD存储
- **GPU**: NVIDIA GPU (支持CUDA 11.0+) - 可选，用于加速
- **网络**: 高速宽带连接

### 软件依赖
- Git
- Docker (可选，用于数据库部署)
- Neo4j Desktop (可选)
- Ollama (可选，用于本地模型)

## Python环境安装

### Windows

#### 方法1: 使用Python官网安装包
1. 访问 [python.org](https://www.python.org/downloads/)
2. 下载Python 3.9或3.10版本
3. 运行安装程序，**勾选"Add Python to PATH"**
4. 验证安装：
```cmd
python --version
pip --version
```

#### 方法2: 使用Anaconda
```cmd
# 下载并安装Anaconda
# 创建专用环境
conda create -n memcube python=3.9
conda activate memcube
```

### macOS

#### 使用Homebrew (推荐)
```bash
# 安装Homebrew (如果尚未安装)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装Python
brew install python@3.9

# 验证安装
python3.9 --version
pip3.9 --version
```

#### 使用pyenv
```bash
# 安装pyenv
brew install pyenv

# 安装Python 3.9
pyenv install 3.9.13

# 设置全局Python版本
pyenv global 3.9.13

# 添加到shell配置文件
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

### Linux (Ubuntu/Debian)

```bash
# 更新包管理器
sudo apt update

# 安装Python 3.9和相关工具
sudo apt install python3.9 python3.9-pip python3.9-venv python3.9-dev

# 安装编译依赖
sudo apt install build-essential curl

# 验证安装
python3.9 --version
pip3.9 --version
```

### Linux (CentOS/RHEL)

```bash
# 安装EPEL仓库
sudo yum install epel-release

# 安装Python 3.9
sudo yum install python39 python39-pip python39-devel

# 验证安装
python3.9 --version
pip3.9 --version
```

## 项目安装

### 1. 克隆项目

```bash
# 使用HTTPS
git clone https://github.com/your-repo/memcube-political.git

# 或使用SSH
git clone git@github.com:your-repo/memcube-political.git

# 进入项目目录
cd memcube-political
```

### 2. 创建虚拟环境

```bash
# 使用venv (推荐)
python3.9 -m venv venv

# 激活虚拟环境

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 验证虚拟环境
which python
# 应该显示项目目录下的python路径
```

### 3. 升级pip和安装基础工具

```bash
# 升级pip到最新版本
pip install --upgrade pip

# 安装wheel (加速后续安装)
pip install wheel

# 安装setuptools
pip install --upgrade setuptools
```

### 4. 安装项目依赖

#### 方法1: 使用requirements.txt
```bash
# 安装所有依赖
pip install -r requirements.txt

# 如果遇到版本冲突，可以尝试
pip install --force-reinstall -r requirements.txt
```

#### 方法2: 分步安装 (解决依赖冲突)
```bash
# 先安装基础依赖
pip install numpy==1.24.3
pip install scipy==1.10.1
pip install scikit-learn==1.2.2

# 再安装机器学习和深度学习库
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.30.0
pip install sentence-transformers==2.2.2

# 安装图数据库客户端
pip install neo4j==5.9.0
pip install qdrant-client==1.3.2

# 安装其他依赖
pip install -r requirements.txt
```

### 5. 验证Python包安装

```bash
# 创建验证脚本
python -c "
import sys
print('Python version:', sys.version)

packages = [
    'numpy', 'scipy', 'networkx', 'matplotlib',
    'neo4j', 'qdrant_client', 'transformers',
    'sentence_transformers', 'pyyaml', 'tqdm'
]

for package in packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError as e:
        print(f'✗ {package}: {e}')
"
```

## 数据库安装

### Neo4j (图数据库)

#### 方法1: 使用Neo4j Desktop (推荐个人开发)

1. 下载 [Neo4j Desktop](https://neo4j.com/download/)
2. 安装并启动Neo4j Desktop
3. 创建新项目，添加数据库
4. 设置用户名和密码 (默认: neo4j / 密码)
5. 启动数据库

#### 方法2: 使用Docker

```bash
# 拉取Neo4j镜像
docker pull neo4j:5.9-community

# 创建并启动容器
docker run \
    --name neo4j-memcube \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/MY_STRONG_PASSWORD \
    neo4j:5.9-community

# 等待启动完成 (约30秒)
docker logs neo4j-memcube
```

#### 方法3: 服务器安装

```bash
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 5.9' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# 启动服务
sudo systemctl start neo4j
sudo systemctl enable neo4j

# 设置密码
sudo cypher-shell -u neo4j
# 然后执行: CALL dbms.security.changePassword('new_password');
```

### Qdrant (向量数据库)

#### 方法1: 使用Docker (推荐)

```bash
# 拉取Qdrant镜像
docker pull qdrant/qdrant:latest

# 创建并启动容器
docker run -d --name qdrant-memcube \
    -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant:latest

# 验证安装
curl http://localhost:6333/collections
```

#### 方法2: 本地二进制安装

```bash
# Linux/macOS
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-linux-x64.tar.gz | tar xz
./qdrant/x86_64-unknown-linux-gnu/qdrant &

# Windows
# 下载 https://github.com/qdrant/qdrant/releases/latest/download/qdrant-windows-x64.exe
# 并运行
```

#### 方法3: 从源码编译

```bash
# 需要Rust环境
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 克隆并编译
git clone https://github.com/qdrant/qdrant.git
cd qdrant
cargo build --release

# 运行
./target/release/qdrant &
```

### Ollama (本地模型 - 可选)

#### 安装Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 下载并运行 https://ollama.ai/download/OllamaSetup.exe

# 启动Ollama服务
ollama serve &
```

#### 下载模型

```bash
# 下载embedding模型
ollama pull bge-m3

# 下载对话模型 (可选)
ollama pull llama2
ollama pull qwen:7b

# 验证安装
ollama list
```

## 配置文件设置

### 1. API密钥配置

创建 `config/api_keys.yaml`:

```yaml
# Gemini API配置 (Google)
gemini:
  api_key: "YOUR_GEMINI_API_KEY_HERE"
  # 获取API密钥: https://makersuite.google.com/app/apikey

# OpenAI API配置 (备用)
openai:
  api_key: "YOUR_OPENAI_API_KEY_HERE"
  organization: "YOUR_ORG_ID"  # 可选
  # 获取API密钥: https://platform.openai.com/api-keys

# 其他API配置
claude:
  api_key: "YOUR_CLAUDE_API_KEY_HERE"

zhipuai:
  api_key: "YOUR_ZHIPU_API_KEY_HERE"
```

### 2. 系统配置

创建 `config/config.yaml`:

```yaml
# MemCube政治理论概念图谱扩增配置

# API配置
api:
  model_thinker: "gemini-2.5-flash"
  model_extractor: "gemini-2.5-flash"
  model_expander: "gemini-2.5-flash"
  model_qa_generator: "gemini-2.5-flash"
  temperature: 0.7
  max_tokens: 32768
  max_retries: 3
  timeout: 60

# 概念扩增配置
concept_expansion:
  similarity_threshold: 0.80
  new_concept_rate_threshold: 0.10
  new_edge_rate_threshold: 0.05
  max_iterations: 10
  batch_size: 50
  max_workers: 10

# 向量化配置
embedding:
  model_name: "bge-m3:567m"
  model_type: "ollama"
  ollama_url: "http://localhost:11434"
  batch_size: 16
  device: "cpu"

# 图数据库配置
graph_database:
  enabled: true
  type: "neo4j"
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "YOUR_NEO4J_PASSWORD"
    database: "neo4j"

# 向量数据库配置
vector_database:
  enabled: true
  type: "qdrant"
  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "political_concepts"
    vector_size: 1024
    distance: "Cosine"

# 数据路径
paths:
  seed_concepts: "data/seed_concepts.txt"
  qa_data: "data/transformed_political_data.json"
  concept_graph_dir: "data/concept_graph"
  results_dir: "results"
  logs_dir: "logs"

# 日志配置
logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
  rotation: "1 day"
  retention: "30 days"
```

### 3. 环境变量配置 (可选)

创建 `.env` 文件:

```bash
# API密钥 (可选，可以放在api_keys.yaml中)
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# 数据库配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama配置
OLLAMA_URL=http://localhost:11434

# 其他配置
PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

## 验证安装

### 1. 创建验证脚本

创建 `verify_installation.py`:

```python
#!/usr/bin/env python3
"""
安装验证脚本
"""

import sys
import os
import importlib

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本满足要求")
        return True
    else:
        print("✗ Python版本不满足要求 (需要3.8+)")
        return False

def check_packages():
    """检查必需的Python包"""
    required_packages = [
        'numpy', 'scipy', 'networkx', 'matplotlib', 'yaml',
        'neo4j', 'qdrant_client', 'transformers', 'sentence_transformers',
        'tqdm', 'requests', 'torch'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)

    return len(missing_packages) == 0, missing_packages

def check_config_files():
    """检查配置文件"""
    config_files = [
        'config/config.yaml',
        'config/api_keys.yaml'
    ]

    missing_files = []

    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files

def check_database_connections():
    """检查数据库连接"""
    try:
        # 检查Neo4j连接
        from neo4j import GraphDatabase
        print("✓ Neo4j客户端库可用")

        # 这里可以尝试连接，但需要配置
        print("ℹ Neo4j连接需要配置文件")

    except ImportError:
        print("✗ Neo4j客户端库不可用")
        return False

    try:
        # 检查Qdrant连接
        from qdrant_client import QdrantClient
        print("✓ Qdrant客户端库可用")
        print("ℹ Qdrant连接需要配置文件")

    except ImportError:
        print("✗ Qdrant客户端库不可用")
        return False

    return True

def check_directories():
    """检查目录结构"""
    directories = [
        'config', 'src', 'data', 'docs', 'logs', 'results'
    ]

    missing_dirs = []

    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ (缺少)")
            missing_dirs.append(directory)

    return len(missing_dirs) == 0, missing_dirs

def main():
    """主验证函数"""
    print("=" * 50)
    print("MemCube Political 安装验证")
    print("=" * 50)

    all_checks_passed = True

    # 检查Python版本
    if not check_python_version():
        all_checks_passed = False

    print()

    # 检查Python包
    print("检查Python包:")
    packages_ok, missing_packages = check_packages()
    if not packages_ok:
        print(f"缺少包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        all_checks_passed = False

    print()

    # 检查配置文件
    print("检查配置文件:")
    config_ok, missing_files = check_config_files()
    if not config_ok:
        print(f"缺少文件: {', '.join(missing_files)}")
        print("请复制相应的.example文件并配置")
        all_checks_passed = False

    print()

    # 检查目录结构
    print("检查目录结构:")
    dirs_ok, missing_dirs = check_directories()
    if not dirs_ok:
        print(f"缺少目录: {', '.join(missing_dirs)}")
        print("请创建缺少的目录")
        all_checks_passed = False

    print()

    # 检查数据库连接
    print("检查数据库连接:")
    if not check_database_connections():
        all_checks_passed = False

    print()

    # 总结
    print("=" * 50)
    if all_checks_passed:
        print("🎉 所有检查通过！安装成功。")
        print("\n下一步:")
        print("1. 配置 config/api_keys.yaml")
        print("2. 启动数据库服务 (Neo4j, Qdrant)")
        print("3. 运行: python main.py")
    else:
        print("❌ 部分检查失败，请按照上述说明修复问题。")
        print("\n如果需要帮助，请查看故障排除部分。")

    print("=" * 50)

    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### 2. 运行验证脚本

```bash
# 确保虚拟环境已激活
python verify_installation.py
```

## 故障排除

### 常见安装问题

#### 1. Python版本问题
```bash
# 错误: Python版本过低
# 解决方案: 安装Python 3.8+

# Ubuntu/Debian
sudo apt install python3.9 python3.9-venv

# CentOS/RHEL
sudo yum install python39
```

#### 2. 包安装失败
```bash
# 错误: 某些包安装失败
# 解决方案: 尝试不同的安装方法

# 方法1: 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 方法2: 分步安装
pip install numpy scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
```

#### 3. 编译错误
```bash
# 错误: 编译某些包时出错
# 解决方案: 安装编译工具

# Ubuntu/Debian
sudo apt install build-essential python3.9-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python39-devel

# Windows
# 安装 Visual Studio Build Tools 或 Visual Studio Community
```

#### 4. 内存不足
```bash
# 错误: 安装时内存不足
# 解决方案: 限制并行安装
pip install --no-cache-dir -r requirements.txt

# 或者逐个安装
pip install numpy
pip install scipy
pip install torch
# ...
```

### 运行时问题

#### 1. API连接问题
```bash
# 错误: API密钥无效或网络连接问题
# 解决方案: 检查配置文件和网络

# 测试网络连接
curl -I https://generativelanguage.googleapis.com

# 检查API密钥格式
# Gemini API密钥应该类似: AIzaSyD...
```

#### 2. 数据库连接问题
```bash
# 错误: 无法连接到数据库
# 解决方案: 检查数据库服务状态

# 检查Neo4j
docker ps | grep neo4j
# 或
systemctl status neo4j

# 检查Qdrant
docker ps | grep qdrant
curl http://localhost:6333/collections
```

#### 3. 权限问题
```bash
# 错误: 文件权限不足
# 解决方案: 修改文件权限

# Linux/macOS
chmod +x scripts/*.sh
chmod -R 755 data/

# Windows
# 以管理员身份运行命令提示符
```

### 获取帮助

如果遇到无法解决的问题，可以：

1. **查看日志文件**: `logs/concept_expansion.log`
2. **检查GitHub Issues**: https://github.com/your-repo/memcube-political/issues
3. **提交新Issue**: 提供详细的错误信息和系统环境
4. **加入社区讨论**: 获取社区支持

## 可选组件

### 1. GPU加速 (可选)

#### NVIDIA GPU设置
```bash
# 安装CUDA Toolkit (根据你的CUDA版本)
# 访问 https://developer.nvidia.com/cuda-downloads

# 安装cuDNN
# 访问 https://developer.nvidia.com/cudnn

# 安装PyTorch GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 配置GPU使用
```yaml
# config/config.yaml
embedding:
  device: "cuda"  # 改为cuda
```

### 2. 开发环境设置

```bash
# 安装开发依赖
pip install pytest black flake8 mypy jupyter

# 安装pre-commit钩子
pip install pre-commit
pre-commit install
```

### 3. Jupyter Notebook支持

```bash
# 安装Jupyter
pip install jupyterlab ipywidgets

# 启动Jupyter
jupyter lab
```

完成以上安装步骤后，您就可以开始使用MemCube Political系统了！