# Installation Guide

## 📋 System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Internet**: Stable connection for API access

### Recommended Requirements
- **Operating System**: Windows 11+, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or higher
- **RAM**: 8GB+ (16GB for large datasets)
- **Storage**: 50GB+ free space (SSD recommended)
- **GPU**: Optional CUDA-compatible GPU
- **Internet**: High-speed connection

### Supported Platforms
| Platform | Status | Notes |
|----------|--------|-------|
| Windows 10/11 | ✅ Fully Supported | Install WSL2 for better performance |
| macOS (Intel) | ✅ Fully Supported | Homebrew recommended |
| macOS (Apple Silicon) | ✅ Fully Supported | Rosetta 2 for some dependencies |
| Ubuntu 18.04+ | ✅ Fully Supported | Native performance |
| CentOS/RHEL 7+ | ✅ Supported | May need additional repositories |
| Debian 10+ | ✅ Supported | Standard installation |

## 🐍 Python Installation

### Windows

#### Option 1: Python.org Installer
1. Download Python 3.9+ from [python.org](https://python.org)
2. Run installer with "Add to PATH" checked
3. Verify installation:
```cmd
python --version
pip --version
```

#### Option 2: Microsoft Store
1. Open Microsoft Store
2. Search for "Python 3.9" or higher
3. Click "Get" to install
4. Verify installation from Command Prompt

#### Option 3: Winget
```cmd
winget install Python.Python.3.11
```

### macOS

#### Option 1: Homebrew (Recommended)
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Add to PATH (add to ~/.zshrc or ~/.bash_profile)
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### Option 2: Python.org Installer
1. Download macOS installer from [python.org](https://python.org)
2. Run installer and follow instructions
3. Verify in Terminal:
```bash
python3 --version
pip3 --version
```

### Linux (Ubuntu/Debian)

#### Option 1: APT Repository
```bash
# Update package lists
sudo apt update

# Install Python and development tools
sudo apt install python3.11 python3.11-dev python3-pip python3-venv

# Verify installation
python3.11 --version
pip3 --version
```

#### Option 2: Dead Snakes PPA (Newer Versions)
```bash
# Add PPA repository
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-dev python3.11-venv

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

### Linux (CentOS/RHEL/Fedora)

#### CentOS/RHEL
```bash
# Enable EPEL repository
sudo yum install epel-release

# Install Python development tools
sudo yum install python3 python3-devel python3-pip

# Install virtual environment
sudo yum install python3-virtualenv
```

#### Fedora
```bash
# Install Python and tools
sudo dnf install python3 python3-devel python3-pip python3-virtualenv
```

## 🦙 Ollama Installation

### Windows

#### Option 1: Download Installer
1. Download Windows installer from [ollama.com/download](https://ollama.com/download)
2. Run installer (.exe)
3. Follow installation prompts
4. Ollama will start automatically and run on startup

#### Option 2: Winget
```cmd
winget install Ollama.Ollama
```

#### Verification
```cmd
# Check Ollama version
ollama --version

# List available models
ollama list

# Test service
curl http://localhost:11434/api/tags
```

### macOS

#### Option 1: Homebrew (Recommended)
```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve
```

#### Option 2: Download Installer
1. Download macOS installer from [ollama.com/download](https://ollama.com/download)
2. Run installer (.pkg)
3. Launch Ollama from Applications

#### Verification
```bash
# Check Ollama version
ollama --version

# Start service if not running
ollama serve

# Test service
curl http://localhost:11434/api/tags
```

### Linux

#### Option 1: Official Install Script (Recommended)
```bash
# Download and install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama  # Start on boot

# Verify installation
ollama --version
```

#### Option 2: Manual Installation
```bash
# Download binary
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/bin/

# Create systemd service
sudo tee /etc/systemd/system/ollama.service > /dev/null <<EOF
[Unit]
Description=Ollama
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

# Create ollama user
sudo useradd -r -s /bin/false -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $USER

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl start ollama
sudo systemctl enable ollama
```

#### Docker Installation
```bash
# Pull Ollama image
docker pull ollama/ollama

# Run container
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Or using docker-compose
cat > docker-compose.yml <<EOF
version: '3.8'
services:
  ollama:
    image: ollama/ollama
    volumes:
      - ollama:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped

volumes:
  ollama:
EOF

docker-compose up -d
```

## 📦 MemCube Political Installation

### Step 1: Clone Repository

#### Using Git (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd memcube-political

# Verify repository contents
ls -la
```

#### Using Download
1. Download ZIP file from repository
2. Extract to desired location
3. Navigate to extracted directory

### Step 2: Create Virtual Environment

#### Windows
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
python --version
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python
python --version
```

### Step 3: Install Dependencies

#### Basic Installation
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
pip list
```

#### Development Installation
```bash
# Install additional development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

#### Troubleshooting Dependencies

**Issue: numpy installation fails on Windows**
```cmd
# Install from precompiled wheels
pip install numpy --only-binary :all:

# Or use conda
conda install numpy
```

**Issue: torch installation fails**
```bash
# CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA version (if NVIDIA GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Issue: transformers installation**
```bash
# Install with specific versions
pip install transformers==4.36.0
pip install tokenizers==0.15.0
```

### Step 4: Download Required Models

#### BGE-M3 Embedding Model
```bash
# Download BGE-M3 model for Ollama
ollama pull bge-m3

# Verify installation
ollama list
# Should show: bge-m3:latest

# Test model
curl http://localhost:11434/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "prompt": "test embedding"
  }'
```

### Step 5: Configuration Setup

#### Copy Configuration Templates
```bash
# Copy example configurations
cp config/api_keys.yaml.example config/api_keys.yaml
cp config/config.yaml.example config/config.yaml

# Verify files were created
ls -la config/
```

#### Edit API Configuration
```bash
# Use your preferred text editor
# Linux/macOS:
nano config/api_keys.yaml
# vim config/api_keys.yaml

# Windows:
notepad config/api_keys.yaml
# code config/api_keys.yaml
```

Add your API keys:
```yaml
# Example for Google Gemini
google:
  api_key: "your-gemini-api-key-here"

# Example for OpenAI
openai:
  api_key: "your-openai-api-key-here"

# Example for custom API
custom:
  api_key: "your-custom-api-key"
  base_url: "https://your-custom-endpoint.com/v1"
```

## ✅ Installation Verification

### Step 1: Environment Check
```bash
# Run comprehensive environment check
python main.py --check-env

# Expected output:
# ✓ Python version: 3.9.x
# ✓ Required packages installed
# ✓ Ollama service running
# ✓ BGE-M3 model available
# ✓ Configuration files exist
```

### Step 2: API Connection Test
```bash
# Test API configuration
python main.py --test-api

# Expected output:
# ✓ Google Gemini API: Connected
# ✓ Ollama embeddings: Connected
```

### Step 3: System Test
```bash
# Run system functionality tests
python scripts/test_system.py

# Expected output:
# ✓ Concept analyzer: Working
# ✓ Embedding client: Working
# ✓ Configuration loading: Working
```

### Step 4: Quick Start Test
```bash
# Run quick start with small dataset
python main.py --quick-start

# Monitor output for successful completion
```

## 🔧 Optional Components

### Database Installation (Optional)

#### Neo4j (Graph Database)
```bash
# Docker installation
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e 'NEO4J_AUTH=neo4j/password' \
  -e 'NEO4J_PLUGINS=["apoc"]' \
  neo4j:latest

# Native installation (Ubuntu/Debian)
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j
```

#### Qdrant (Vector Database)
```bash
# Docker installation
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  qdrant/qdrant

# Native installation
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-linux-x86_64.tar.gz | tar xz
./qdrant/x86_64-unknown-linux-gnu/qdrant
```

#### ChromaDB (Vector Database)
```bash
# Install Python client
pip install chromadb

# Install standalone server
pip install chromadb
python -c "import chromadb; chromadb.Client()"
```

### GPU Support (Optional)

#### NVIDIA GPU Setup
```bash
# Install CUDA toolkit
# Visit https://developer.nvidia.com/cuda-downloads

# Install cuDNN
# Visit https://developer.nvidia.com/cudnn

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(torch.cuda.is_available())"
```

#### AMD GPU Support
```bash
# Install ROCm (Linux only)
# Follow instructions at https://rocm.docs.amd.com/en/latest/deploy/linux/index.html

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## 🐛 Troubleshooting Installation

### Common Issues

#### Python Installation Issues
**Problem**: "Python not found" or version too old
```bash
# Check if Python is installed
python --version
python3 --version

# Install or upgrade Python following platform-specific instructions above
```

**Problem**: Virtual environment creation fails
```bash
# Ensure python-venv is installed (Linux)
sudo apt install python3-venv

# Try alternative creation method
python -m pip install virtualenv
virtualenv venv
```

#### Ollama Installation Issues
**Problem**: "Permission denied" or service won't start
```bash
# Linux: Check user permissions
sudo usermod -a -G ollama $USER
newgrp ollama

# Check service status
sudo systemctl status ollama
sudo journalctl -u ollama -f
```

**Problem**: Model download fails
```bash
# Check network connection
curl -I https://ollama.com

# Manually download model
ollama pull bge-m3 --verbose

# Clear cache and retry
rm -rf ~/.ollama/models/manifests/registry.ollama.ai/library/bge-m3
ollama pull bge-m3
```

#### Dependency Installation Issues
**Problem**: "Failed building wheel" for certain packages
```bash
# Install build tools (Ubuntu/Debian)
sudo apt install build-essential python3-dev

# Install build tools (Windows)
# Install Visual Studio Build Tools or Visual Studio Community

# Install build tools (macOS)
xcode-select --install
```

**Problem**: Package conflicts
```bash
# Create clean virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install with specific versions
pip install -r requirements.txt --force-reinstall
```

### Platform-Specific Issues

#### Windows
**Problem**: Long path names cause errors
```cmd
# Enable long path support
# Run as Administrator and execute:
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Problem**: DLL loading errors
```cmd
# Install Microsoft Visual C++ Redistributable
# Download from: https://docs.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
```

#### macOS
**Problem**: Python installation via Homebrew issues
```bash
# Update Homebrew
brew update
brew doctor

# Reinstall Python
brew reinstall python@3.11
```

**Problem**: Permission issues with Ollama
```bash
# Fix permissions
sudo chown -R $USER:$(id -gn $USER) ~/.ollama
```

#### Linux
**Problem**: Package manager conflicts
```bash
# Fix broken packages
sudo apt --fix-broken install

# Update package lists
sudo apt update && sudo apt upgrade
```

## 📞 Getting Help

### Installation Support Resources
- **GitHub Issues**: Report installation bugs
- **Documentation**: Check latest installation guides
- **Community Forums**: Get help from other users
- **Discord/Slack**: Real-time support chat

### What to Include in Support Requests
1. Operating system and version
2. Python version
3. Error messages (full output)
4. Steps taken so far
5. Configuration files (sanitized)

### Diagnostic Information Collection
```bash
# Generate diagnostic report
python scripts/generate_diagnostics.py

# Manually collect information
python --version
pip list
ollama --version
curl -s http://localhost:11434/api/tags
```

---

Once installation is complete and verified, proceed to the [User Manual](USER_MANUAL.md) for usage instructions.