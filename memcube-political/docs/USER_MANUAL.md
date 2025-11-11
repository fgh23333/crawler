# User Manual

## 📖 Table of Contents

1. [Getting Started](#-getting-started)
2. [System Overview](#-system-overview)
3. [Installation](#-installation)
4. [Configuration](#-configuration)
5. [Running the System](#-running-the-system)
6. [Advanced Usage](#-advanced-usage)
7. [Troubleshooting](#-troubleshooting)
8. [Best Practices](#-best-practices)

## 🚀 Getting Started

### Quick Start Guide

For users who want to get running immediately:

```bash
# 1. Clone and setup
git clone <repository-url>
cd memcube-political
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Ollama
# Install from https://ollama.com/download
ollama pull bge-m3
ollama serve

# 4. Configure APIs
cp config/api_keys.yaml.example config/api_keys.yaml
# Edit the file with your API keys

# 5. Run the system
python main.py --quick-start
```

### System Requirements

#### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Storage**: 10GB free space
- **Network**: Stable internet connection

#### Recommended Requirements
- **Python**: 3.9 or higher
- **RAM**: 8GB or more
- **Storage**: 50GB free space
- **Network**: High-speed internet connection
- **GPU**: Optional, for accelerated processing

#### Supported Operating Systems
- **Windows**: 10/11 (64-bit)
- **macOS**: 10.15+ (Intel/Apple Silicon)
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+

## 🎯 System Overview

### What MemCube Political Does

MemCube Political is an AI-powered system that:

1. **Analyzes Political Theory Concepts**: Uses advanced language models to understand political theory concepts deeply
2. **Builds Knowledge Graphs**: Automatically discovers relationships between concepts
3. **Generates Educational Content**: Creates high-quality question-answer pairs for learning
4. **Assesses Quality**: Provides comprehensive evaluation of generated content

### Core Workflow

```
Seed Concepts → Analysis → Extraction → Graph Building → Q&A Generation → Quality Assessment
```

### Key Components

- **Concept Analyzer**: Deep semantic analysis of political concepts
- **Concept Extractor**: Identifies and extracts key concepts from text
- **Graph Builder**: Constructs and expands knowledge graphs
- **Q&A Generator**: Creates educational content from the knowledge graph
- **Quality Evaluator**: Assesses the quality of all outputs

## 🛠️ Installation

### Step 1: Prerequisites

#### Python Installation
```bash
# Check Python version (must be 3.8+)
python --version
# or
python3 --version

# If not installed, download from https://python.org
```

#### Ollama Installation
```bash
# macOS
brew install ollama

# Windows (Download from https://ollama.com/download)

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Download BGE-M3 model
ollama pull bge-m3
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd memcube-political

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Configuration Setup

```bash
# Copy configuration templates
cp config/api_keys.yaml.example config/api_keys.yaml
cp config/config.yaml.example config/config.yaml

# Edit configuration files
# Use your preferred text editor
nano config/api_keys.yaml
# or
notepad config/api_keys.yaml
```

### Step 4: Verify Installation

```bash
# Check environment
python main.py --check-env

# Test API connections
python main.py --test-api

# Run system tests
python scripts/test_system.py
```

## ⚙️ Configuration

### API Configuration (`config/api_keys.yaml`)

This file contains your API keys and service endpoints:

```yaml
# OpenAI Configuration (if using OpenAI models)
openai:
  api_key: "your-openai-api-key-here"
  base_url: "https://api.openai.com/v1"  # Optional: custom endpoint
  organization: "your-org-id"  # Optional

# Google Gemini Configuration (if using Google models)
google:
  api_key: "your-gemini-api-key-here"
  base_url: "https://generativelanguage.googleapis.com"  # Optional

# Custom OpenAI-compatible API (for other providers)
custom:
  api_key: "your-custom-api-key"
  base_url: "https://your-custom-endpoint.com/v1"
```

#### Supported API Providers

1. **OpenAI**: GPT-3.5, GPT-4, GPT-4o models
2. **Google**: Gemini models (Flash, Pro)
3. **Anthropic**: Claude models (via OpenAI-compatible endpoint)
4. **Custom**: Any OpenAI-compatible API endpoint

### Main Configuration (`config/config.yaml`)

#### API Model Configuration
```yaml
api:
  # Models for different tasks
  model_thinker: "gemini-2.5-flash"        # Concept analysis
  model_extractor: "gemini-2.5-flash"     # Concept extraction
  model_expander: "gemini-2.5-flash"      # Graph expansion
  model_qa_generator: "gemini-2.5-flash"  # Q&A generation

  # API settings
  temperature: 0.7
  max_tokens: 32768
  max_retries: 3
  timeout: 60

  # Proxy settings (optional)
  proxy:
    http: "http://127.0.0.1:7890"
    https: "http://127.0.0.1:7890"
```

#### Embedding Configuration
```yaml
embedding:
  model_name: "bge-m3"
  model_type: "ollama"
  ollama_url: "http://localhost:11434"
  batch_size: 16
  device: "cpu"  # or "cuda" if available
```

#### Concept Expansion Configuration
```yaml
concept_expansion:
  similarity_threshold: 0.80        # Min similarity for connections
  new_concept_rate_threshold: 0.10  # Min growth rate to continue
  new_edge_rate_threshold: 0.05     # Min edge growth rate
  max_iterations: 10                # Maximum expansion rounds
  batch_size: 50                    # Concepts per batch
  max_workers: 10                   # Parallel processing
```

#### Q&A Generation Configuration
```yaml
qa_generation:
  concepts_per_batch: 20            # Concepts per processing batch
  qa_pairs_per_concept: 3           # Questions per concept
  qa_pairs_per_concept_pair: 2      # Questions per concept pair
  max_workers: 5                    # Parallel Q&A generation
  enable_validity_check: true       # Enable quality filtering
  min_validity_threshold: 0.6       # Minimum quality score
```

#### Database Configuration
```yaml
graph_database:
  enabled: true
  type: "neo4j"  # neo4j, arangodb, janusgraph

  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "your-password"
    database: "neo4j"

vector_database:
  enabled: true
  type: "qdrant"  # qdrant, chroma, faiss, milvus

  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "political_concepts"
    vector_size: 1024
```

## 🏃 Running the System

### Basic Commands

#### Quick Start
```bash
# Run with default settings
python main.py --quick-start
```

#### Stage-based Execution
```bash
# Run only concept graph expansion
python main.py --stage concept-expansion

# Run only Q&A generation (requires existing graph)
python main.py --stage qa-generation

# Run complete pipeline
python main.py --stage all
```

#### Testing and Validation
```bash
# Check environment and dependencies
python main.py --check-env

# Test API configuration
python main.py --test-api

# Run system validation
python scripts/test_system.py
```

### Command Line Options

#### Main Entry Point (`main.py`)
```bash
# Usage: python main.py [OPTIONS]

Options:
  --stage STAGE          Processing stage: concept-expansion, qa-generation, all
  --quick-start          Run with default settings
  --check-env           Validate environment
  --test-api           Test API connections
  --config CONFIG       Custom config file path
  --verbose            Enable detailed logging
  --dry-run            Show what would be executed without running
  --help               Show help message
```

#### Script Options

##### Environment Check (`scripts/check_env.py`)
```bash
python scripts/check_env.py
# Validates:
# - Python version
# - Required packages
# - Ollama service
# - API configuration
```

##### Quick Start (`scripts/quick_start.py`)
```bash
python scripts/quick_start.py
# Runs complete pipeline with optimized settings
```

### Monitoring Progress

#### Real-time Logs
```bash
# Follow logs in real-time
tail -f logs/memcube_$(date +%Y-%m-%d).log

# Show only important messages
tail -f logs/memcube_*.log | grep -E "(INFO|WARNING|ERROR)"
```

#### Progress Indicators
The system provides real-time progress updates:
- Processing stage completion
- Batch processing progress
- Error and warning messages
- Performance metrics

### Output Files

#### Concept Graph Outputs (`data/concept_graph/`)
```
final_concept_graph.json      # Complete knowledge graph
convergence_history.json      # Expansion convergence data
expansion_summary.json        # Statistics and metrics
iteration_results/           # Per-iteration outputs
```

#### Q&A Generation Outputs (`results/`)
```
political_theory_qa_dataset.json    # Complete Q&A dataset
political_theory_qa_training.jsonl  # Training format data
qa_generation_summary.json          # Generation statistics
quality_reports/                    # Quality assessment reports
```

## 🔧 Advanced Usage

### Custom Seed Concepts

#### Using Your Own Concepts
```bash
# Create custom concepts file
echo "democracy" > data/my_concepts.txt
echo "constitutional law" >> data/my_concepts.txt
echo "civil rights" >> data/my_concepts.txt

# Update configuration to use custom concepts
# Edit config/config.yaml:
paths:
  seed_concepts: "data/my_concepts.txt"
```

#### Concept File Format
```
# Each line should contain one concept
democracy
constitutional law
civil rights
political philosophy
social contract
```

### Custom Processing Parameters

#### High-Performance Configuration
```yaml
# For systems with abundant resources
concept_expansion:
  max_workers: 20
  batch_size: 100
  similarity_threshold: 0.75  # Lower threshold for more connections

qa_generation:
  max_workers: 10
  concepts_per_batch: 50
  qa_pairs_per_concept: 5
```

#### Resource-Conserving Configuration
```yaml
# For systems with limited resources
concept_expansion:
  max_workers: 3
  batch_size: 20
  similarity_threshold: 0.85  # Higher threshold for quality

qa_generation:
  max_workers: 2
  concepts_per_batch: 10
  qa_pairs_per_concept: 2
```

### Database Integration

#### Using External Databases
```yaml
# Neo4j setup
graph_database:
  enabled: true
  type: "neo4j"
  neo4j:
    uri: "bolt://your-neo4j-host:7687"
    username: "neo4j"
    password: "your-password"
    database: "political_concepts"

# Qdrant setup
vector_database:
  enabled: true
  type: "qdrant"
  qdrant:
    host: "your-qdrant-host"
    port: 6333
    api_key: "your-api-key"  # If required
```

#### Memory-Only Mode
```yaml
graph_database:
  enabled: false

vector_database:
  enabled: false
```

### Quality Customization

#### Adjusting Quality Thresholds
```yaml
concept_validation:
  validity_threshold: 0.8        # Concept quality threshold
  quality_filters:
    max_concept_length: 10       # Maximum words per concept
    min_concept_length: 1        # Minimum words per concept
    exclude_numbers: true        # Filter out numeric concepts
    exclude_duplicates: true     # Remove duplicate concepts

qa_generation:
  enable_validity_check: true
  min_validity_threshold: 0.7    # Q&A quality threshold
  max_question_length: 200       # Maximum question characters
  min_answer_length: 50          # Minimum answer characters
```

## 🚨 Troubleshooting

### Common Issues

#### API Connection Problems

**Symptom**: "API connection failed" or authentication errors
```bash
# Test API configuration
python main.py --test-api

# Common solutions:
# 1. Verify API keys in config/api_keys.yaml
# 2. Check internet connection
# 3. Verify API service status
# 4. Check rate limits
```

#### Ollama Service Issues

**Symptom**: "Cannot connect to Ollama service" or embedding errors
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Reinstall BGE-M3 model
ollama pull bge-m3

# Common solutions:
# 1. Ensure Ollama is running
# 2. Verify BGE-M3 model is installed
# 3. Check port 11434 is available
# 4. Restart Ollama service
```

#### Memory Issues

**Symptom**: "Out of memory" or system slowdown
```bash
# Reduce batch sizes
# Edit config/config.yaml:
concept_expansion:
  batch_size: 10
  max_workers: 2

qa_generation:
  concepts_per_batch: 5
  max_workers: 1

# Monitor memory usage
python scripts/monitor_resources.py
```

#### Quality Issues

**Symptom**: Low-quality concepts or Q&A pairs
```bash
# Increase quality thresholds
# Edit config/config.yaml:
concept_expansion:
  similarity_threshold: 0.85  # Higher for better quality

qa_generation:
  min_validity_threshold: 0.8  # Higher for better quality

# Enable additional quality checks
concept_validation:
  quality_filters:
    exclude_duplicates: true
    min_concept_length: 2
```

### Error Messages Explained

#### API Errors
- `401 Unauthorized`: Invalid API key
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: API service issue
- `Timeout`: Request took too long

#### System Errors
- `ModuleNotFoundError`: Missing dependencies
- `FileNotFoundError`: Missing data files
- `PermissionError`: File access issues
- `ConnectionError`: Network connectivity problems

### Getting Help

#### Log Analysis
```bash
# Find recent errors
grep -i error logs/memcube_*.log | tail -20

# Find warnings
grep -i warning logs/memcube_*.log | tail -20

# Full error context
grep -A 5 -B 5 "ERROR" logs/memcube_*.log
```

#### Debug Mode
```bash
# Run with verbose logging
python main.py --stage all --verbose

# Run in dry-run mode
python main.py --stage all --dry-run
```

#### Community Support
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest documentation updates
- **Community Forums**: Get help from other users

## 💡 Best Practices

### Performance Optimization

#### System Resources
- Close unnecessary applications during processing
- Use SSD storage for faster I/O operations
- Ensure stable internet connection for API calls
- Monitor system resources during operation

#### Configuration Tuning
- Start with conservative settings, gradually increase
- Monitor quality metrics vs. processing speed
- Adjust batch sizes based on available RAM
- Balance between speed and quality requirements

### Data Management

#### Backup Strategies
```bash
# Back up important data
cp -r data/concept_graph/ backup/concept_graph_$(date +%Y%m%d)/
cp -r results/ backup/results_$(date +%Y%m%d)/

# Back up configuration
cp config/config.yaml backup/config_$(date +%Y%m%d).yaml
```

#### Storage Organization
```
project_data/
├── raw_data/          # Original seed concepts
├── processed_data/    # Generated graphs and Q&A
├── backups/          # Historical backups
└── exports/          # Final formatted outputs
```

### Quality Assurance

#### Validation Checks
- Review sample outputs for quality
- Check concept coverage and diversity
- Validate Q&A accuracy and relevance
- Monitor consistency across runs

#### Iterative Improvement
- Start with small test datasets
- Evaluate quality before scaling up
- Adjust parameters based on results
- Document successful configurations

### Security Considerations

#### API Key Protection
- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Monitor API usage and costs

#### Data Privacy
- Understand data usage policies of API providers
- Consider local processing for sensitive content
- Implement access controls for generated data
- Follow data retention best practices

---

This manual provides comprehensive guidance for users at all levels. For specific technical details, refer to the API documentation and developer guides.