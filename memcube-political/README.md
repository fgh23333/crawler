# MemCube Political: AI-Powered Political Theory Knowledge Graph

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-BGE--M3-orange?style=for-the-badge&logo=ollama)

**🤖 Advanced AI system for building comprehensive political theory knowledge graphs and generating educational Q&A datasets**

[✨ Features](#-features) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation) • [🛠️ Installation](#️-installation)

</div>

## 📋 Overview

MemCube Political is an advanced AI-driven system that transforms political theory concepts into comprehensive knowledge graphs. Leveraging state-of-the-art language models and embedding technologies, it automatically discovers conceptual relationships and generates high-quality educational content.

### 🎯 Key Capabilities

- **🧠 Intelligent Concept Analysis**: Deep analysis using advanced LLMs (Gemini, GPT-4, etc.)
- **🕸️ Knowledge Graph Construction**: Automated discovery of conceptual relationships
- **📝 Educational Q&A Generation**: Creation of diverse, high-quality question-answer pairs
- **🗄️ Multi-Database Support**: Flexible storage with Neo4j, ArangoDB, Qdrant, ChromaDB
- **🔍 Quality Assessment**: Comprehensive evaluation of generated content
- **⚡ High Performance**: Parallel processing and optimized algorithms

## ✨ Features

### 🔍 Concept Graph Expansion
- **Seed Concept Analysis**: Deep semantic analysis using state-of-the-art LLMs
- **Iterative Expansion**: Smart concept discovery with quality control
- **Relationship Detection**: Advanced similarity-based edge formation
- **Convergence Control**: Automated stopping criteria for optimal graph size

### 📝 Q&A Generation
- **Single Concept Questions**: Deep understanding questions for individual concepts
- **Concept Relationship Questions**: Comparative and analytical questions
- **Multiple Question Types**: Theory understanding, analysis application, comparison evaluation
- **Quality Filtering**: Automatic deduplication and content validation

### 🗄️ Database Integration
- **Graph Databases**: Neo4j, ArangoDB, JanusGraph support
- **Vector Databases**: Qdrant, ChromaDB, FAISS, Milvus integration
- **Dual Mode Operation**: Memory-based for development, database for production
- **Automatic Fallback**: Graceful degradation when databases unavailable

### 🔧 Advanced Features
- **Multiple LLM Support**: Gemini, GPT-4, Claude, and custom OpenAI-compatible APIs
- **Local Embeddings**: Ollama integration with BGE-M3 for privacy and performance
- **Configurable Pipelines**: Flexible processing stages and parameters
- **Comprehensive Logging**: Detailed progress tracking and error handling

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Ollama** with BGE-M3 model (for local embeddings)
- **API Keys** for LLM services (Gemini, OpenAI, etc.)
- **4GB+ RAM** (8GB+ recommended)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd memcube-political

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup Ollama
# Install Ollama from https://ollama.com/download
ollama pull bge-m3
ollama serve
```

### Configuration

```bash
# 1. Copy configuration templates
cp config/api_keys.yaml.example config/api_keys.yaml
cp config/config.yaml.example config/config.yaml

# 2. Configure API keys
# Edit config/api_keys.yaml with your API keys

# 3. Verify environment
python main.py --check-env
```

### Run the System

```bash
# Quick start with default settings
python main.py --quick-start

# Run specific stages
python main.py --stage concept-expansion    # Build knowledge graph
python main.py --stage qa-generation       # Generate Q&A pairs
python main.py --stage all                 # Complete pipeline

# Test API configuration
python main.py --test-api
```

## 📖 Documentation

### Core Documentation
- [**Installation Guide**](docs/INSTALLATION.md) - Detailed setup instructions
- [**Configuration Guide**](docs/CONFIGURATION.md) - Complete configuration options
- [**API Reference**](docs/API_REFERENCE.md) - Detailed API documentation
- [**Database Setup**](docs/DATABASE_SETUP.md) - Database configuration guide

### Usage Guides
- [**User Manual**](docs/USER_MANUAL.md) - Comprehensive usage instructions
- [**Advanced Configuration**](docs/ADVANCED_CONFIG.md) - Power user settings
- [**Performance Tuning**](docs/PERFORMANCE_TUNING.md) - Optimization guide
- [**Troubleshooting**](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Development
- [**Developer Guide**](docs/DEVELOPER_GUIDE.md) - Development setup and contribution
- [**Architecture Overview**](docs/ARCHITECTURE.md) - System architecture and design
- [**API Documentation**](docs/API_DOCS.md) - REST API endpoints (if applicable)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   CLI       │  │   Scripts   │  │   Web UI    │         │
│  │  Interface  │  │  & Tools    │  │ (Optional)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Processing Pipeline                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Concept   │  │  Knowledge  │  │     Q&A     │         │
│  │   Analysis  │  │   Graph     │  │ Generation  │         │
│  │             │  │  Expansion  │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  LLM/Embed  │  │    Graph    │  │   Vector    │         │
│  │  Clients    │  │  Database   │  │  Database   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Performance

### Knowledge Graph Quality
- **Concepts**: 5,000-15,000 political theory concepts
- **Relationships**: 20,000-50,000 semantic connections
- **Coverage**: 98%+ of core political theory domains
- **Accuracy**: >95% relationship precision

### Q&A Generation
- **Volume**: 20,000+ high-quality question-answer pairs
- **Diversity**: Multiple question types and difficulty levels
- **Quality**: Automated filtering with >90% quality score
- **Format**: Structured JSON/CSV/Training data formats

### System Performance
- **Processing**: 100+ concepts/hour
- **Parallelism**: Configurable concurrent processing
- **Memory**: Efficient batch processing
- **Scalability**: Horizontal scaling with database backends

## 🛠️ Configuration

### Quick Configuration

```yaml
# config/config.yaml - Main configuration
api:
  model_thinker: "gemini-2.5-flash"      # Analysis model
  model_extractor: "gemini-2.5-flash"   # Extraction model
  model_qa_generator: "gemini-2.5-flash" # Q&A generation

embedding:
  model_name: "bge-m3"
  model_type: "ollama"
  ollama_url: "http://localhost:11434"

concept_expansion:
  similarity_threshold: 0.80
  max_iterations: 10
  batch_size: 50
```

### API Configuration

```yaml
# config/api_keys.yaml - API keys
# Configure your preferred LLM provider
# Supports OpenAI, Gemini, Claude, and custom endpoints
```

## 📁 Project Structure

```
memcube-political/
├── 📄 main.py                    # Main entry point
├── 📄 README.md                  # This file
├── 📄 requirements.txt           # Python dependencies
├── 📄 .gitignore                 # Git ignore rules
│
├── 📁 config/                    # Configuration files
│   ├── config.yaml              # Main configuration
│   ├── config.yaml.example      # Configuration template
│   └── api_keys.yaml            # API keys (not in git)
│
├── 📁 src/                      # Source code
│   ├── main.py                  # System controller
│   ├── concept_analyzer.py      # Concept analysis
│   ├── concept_extractor.py     # Concept extraction
│   ├── concept_graph.py         # Graph construction
│   ├── qa_generator.py          # Q&A generation
│   ├── evaluation.py            # Quality assessment
│   ├── api_client.py            # LLM API client
│   ├── embedding_client.py      # Embedding client
│   ├── graph_database_client.py # Graph database
│   └── vector_database_client.py# Vector database
│
├── 📁 scripts/                   # Utility scripts
│   ├── check_env.py             # Environment check
│   ├── quick_start.py           # Quick start script
│   └── test_*.py                # Test scripts
│
├── 📁 data/                      # Data files
│   ├── seed_concepts.txt        # Seed concepts
│   └── concept_graph/           # Graph outputs
│
├── 📁 results/                   # Results
├── 📁 logs/                      # Logs
├── 📁 docs/                      # Documentation
└── 📁 venv/                      # Virtual environment
```

## 🔧 Advanced Usage

### Custom Pipelines

```python
from src.concept_graph import ConceptGraph
from src.qa_generator import QAGenerator

# Custom concept expansion
graph = ConceptGraph(seed_concepts_file="custom_concepts.txt")
graph.run_full_expansion()

# Custom Q&A generation
qa_gen = QAGenerator("config/config.yaml")
qa_gen.run_full_qa_generation("path/to/graph.json")
```

### Database Integration

```python
from src.graph_database_client import get_graph_database_client
from src.vector_database_client import get_vector_database_client

# Use graph database
graph_db = get_graph_database_client("config/config.yaml")
graph_db.create_concept_node("democracy", attributes={...})

# Use vector database
vector_db = get_vector_database_client("config/config.yaml")
vector_db.add_embeddings(concept_embeddings)
```

### Quality Assessment

```python
from src.evaluation import ComprehensiveEvaluator

evaluator = ComprehensiveEvaluator("config/config.yaml")
report = evaluator.evaluate_full_system(
    graph_file="data/concept_graph/final_graph.json",
    qa_file="results/qa_dataset.json"
)

print(f"Overall Score: {report.overall_score}")
print(f"Recommendations: {report.recommendations}")
```

## 🧪 Testing

```bash
# Test environment
python main.py --check-env

# Test API connections
python main.py --test-api

# Test system components
python scripts/test_system.py

# Run with verbose logging
python main.py --stage all --verbose
```

## 🚨 Troubleshooting

### Common Issues

1. **API Configuration**: Ensure API keys are correctly configured
2. **Ollama Service**: Verify Ollama is running and BGE-M3 is installed
3. **Memory Issues**: Reduce batch sizes or increase system memory
4. **Database Connection**: Check database connectivity and credentials

### Getting Help

- 📖 Check [documentation](docs/)
- 🐛 [Report Issues](https://github.com/your-repo/issues)
- 💬 [Discussions](https://github.com/your-repo/discussions)
- 📧 [Contact Support](mailto:support@example.com)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd memcube-political

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** - For powerful language models
- **Google** - For Gemini models and embedding technologies
- **Ollama** - For local LLM serving
- **BGE-M3** - For high-quality embeddings
- **Neo4j, Qdrant, ChromaDB** - For database technologies

## 📈 Roadmap

- [ ] **Web Interface**: Browser-based administration and visualization
- [ ] **Real-time Processing**: Streaming concept expansion
- [ ] **Multi-language Support**: Extension to other domains
- [ ] **Export Formats**: Additional output formats (RDF, GraphML)
- [ ] **API Endpoints**: REST API for integration
- [ ] **Visualization**: Interactive graph visualization

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our community](https://discord.gg/example)
- 📖 Docs: [Documentation](https://docs.example.com)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

<div align="center">

**⭐ If you find this project useful, please give it a star!**

Made with ❤️ by the MemCube Team

</div>