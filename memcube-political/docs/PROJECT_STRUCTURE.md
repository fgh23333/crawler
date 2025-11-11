# Project Structure Documentation

## 📁 Directory Structure

```
memcube-political/
├── 📄 main.py                           # Main application entry point
├── 📄 README.md                         # Project documentation
├── 📄 requirements.txt                  # Python dependencies
├── 📄 .gitignore                         # Git ignore configuration
│
├── 📁 config/                           # Configuration files
│   ├── config.yaml                     # Main system configuration
│   ├── config.yaml.example             # Configuration template
│   └── api_keys.yaml                   # API keys (not in version control)
│
├── 📁 src/                             # Source code modules
│   ├── __init__.py
│   ├── main.py                         # System controller and orchestration
│   ├── concept_analyzer.py             # Concept analysis with LLMs
│   ├── concept_extractor.py            # Concept extraction from text
│   ├── concept_graph.py                # Graph construction and expansion
│   ├── qa_generator.py                 # Q&A generation from concepts
│   ├── evaluation.py                   # Quality assessment and metrics
│   ├── api_client.py                   # LLM API client (OpenAI-compatible)
│   ├── embedding_client.py             # Embedding client (Ollama)
│   ├── graph_database_client.py        # Graph database abstraction
│   ├── vector_database_client.py       # Vector database abstraction
│   ├── knowledge_base_builder.py       # Knowledge base construction
│   └── prompt_templates.py             # LLM prompt templates
│
├── 📁 scripts/                         # Utility and management scripts
│   ├── check_env.py                    # Environment validation
│   ├── quick_start.py                  # Quick start (memory mode)
│   ├── quick_start_database.py        # Quick start (database mode)
│   ├── test_api_simple.py              # Basic API connectivity test
│   ├── test_api_config.py              # Detailed API configuration test
│   └── test_system.py                  # System functionality tests
│
├── 📁 data/                            # Data storage
│   ├── seed_concepts.txt               # Initial concept seeds (11,027)
│   ├── seed_concepts.json              # Seeds in JSON format
│   ├── transformed_political_data.json # Existing Q&A data (12,092)
│   ├── political_theory_knowledge_base.yaml # Knowledge base
│   └── concept_graph/                  # Generated graph outputs
│
├── 📁 results/                         # Processing results
│   ├── political_theory_qa_dataset.json # Generated Q&A dataset
│   ├── political_theory_qa_training.jsonl # Training format data
│   └── evaluation_reports/             # Quality assessment reports
│
├── 📁 logs/                            # Application logs
├── 📁 docs/                            # Documentation
└── 📁 venv/                            # Python virtual environment
```

## 🔧 Module Responsibilities

### Main Entry Points

#### `main.py` (Root Level)
- **Purpose**: Primary CLI interface and application launcher
- **Features**:
  - Command-line argument parsing
  - Environment validation
  - Stage-based execution control
  - Integration with both memory and database modes

#### `src/main.py`
- **Purpose**: System orchestration and pipeline control
- **Features**:
  - Pipeline stage management
  - Component initialization
  - Error handling and recovery
  - Progress tracking and logging

### Core Processing Modules

#### `concept_analyzer.py`
- **Purpose**: Deep semantic analysis of political theory concepts
- **Responsibilities**:
  - LLM-based concept analysis
  - Structured information extraction
  - Quality assurance of analysis results
  - Batch processing optimization

#### `concept_extractor.py`
- **Purpose**: Extract core concepts from analyzed text
- **Features**:
  - Named entity recognition
  - Concept normalization
  - Duplicate detection
  - Quality scoring

#### `concept_graph.py`
- **Purpose**: Knowledge graph construction and expansion
- **Algorithms**:
  - Iterative concept expansion
  - Similarity-based edge creation
  - Convergence detection
  - Graph quality assessment

#### `qa_generator.py`
- **Purpose**: Generate educational Q&A pairs from concepts
- **Capabilities**:
  - Single concept questions
  - Concept relationship questions
  - Multiple question types
  - Quality filtering and deduplication

#### `evaluation.py`
- **Purpose**: Comprehensive quality assessment
- **Metrics**:
  - Graph structure analysis
  - Semantic quality evaluation
  - Q&A content assessment
  - Performance benchmarking

### Database Abstraction Layer

#### `graph_database_client.py`
- **Purpose**: Graph database operations abstraction
- **Supported Databases**:
  - Neo4j (primary)
  - ArangoDB (alternative)
  - JanusGraph (alternative)
- **Features**:
  - Automatic fallback to memory mode
  - Connection pooling
  - Batch operations
  - Query optimization

#### `vector_database_client.py`
- **Purpose**: Vector database operations for similarity search
- **Supported Databases**:
  - Qdrant (primary)
  - ChromaDB (alternative)
  - FAISS (in-memory)
  - Milvus (distributed)
- **Features**:
  - Embedding storage and retrieval
  - Similarity search optimization
  - Index management
  - Automatic scaling

### API and Integration Layer

#### `api_client.py`
- **Purpose**: Unified LLM API client
- **Supported Providers**:
  - OpenAI (GPT-3.5, GPT-4, GPT-4o)
  - Google (Gemini models)
  - Anthropic (Claude)
  - Custom OpenAI-compatible endpoints
- **Features**:
  - Automatic retry logic
  - Rate limiting
  - Cost tracking
  - Response validation

#### `embedding_client.py`
- **Purpose**: Text embedding generation
- **Integration**:
  - Ollama (BGE-M3)
  - OpenAI embeddings
  - Local transformer models
- **Features**:
  - Batch processing
  - Caching
  - Dimension management
  - Performance optimization

## 🔄 Data Flow Architecture

```
Seed Concepts Input
        ↓
   Concept Analysis
   (LLM Processing)
        ↓
   Concept Extraction
   (NLP Processing)
        ↓
   Graph Construction
   (Similarity Analysis)
        ↓
   Iterative Expansion
   (Quality Control)
        ↓
   Q&A Generation
   (Content Creation)
        ↓
   Quality Assessment
   (Evaluation)
        ↓
   Output Generation
   (Multiple Formats)
```

## 🗄️ Database Integration Strategy

### Dual-Mode Design

#### Memory Mode (Development/Testing)
- **Graph Storage**: NetworkX in-memory graphs
- **Vector Storage**: NumPy arrays
- **Advantages**: Fast, no external dependencies
- **Use Case**: Development, testing, small datasets

#### Database Mode (Production)
- **Graph Storage**: Neo4j/ArangoDB clusters
- **Vector Storage**: Qdrant/ChromaDB clusters
- **Advantages**: Scalable, persistent, concurrent access
- **Use Case**: Production, large datasets, multi-user

### Automatic Failover
- System detects database availability
- Graceful degradation to memory mode
- Seamless operation without user intervention
- Consistent API across modes

## ⚙️ Configuration System

### Hierarchical Configuration

1. **Default Values**: Built-in defaults
2. **Config Files**: `config.yaml` and `api_keys.yaml`
3. **Environment Variables**: Runtime overrides
4. **Command Line**: Immediate parameter changes

### Configuration Categories

#### API Configuration
- Model selection
- Rate limits
- Authentication
- Endpoint configuration

#### Processing Configuration
- Batch sizes
- Concurrency limits
- Quality thresholds
- Convergence parameters

#### Database Configuration
- Connection settings
- Performance tuning
- Backup strategies
- Security options

## 🧪 Testing Architecture

### Test Categories

#### Unit Tests
- Individual module functionality
- Mock external dependencies
- Fast execution
- Comprehensive coverage

#### Integration Tests
- End-to-end workflows
- Database interactions
- API integrations
- Real data validation

#### Performance Tests
- Load testing
- Memory profiling
- Scalability validation
- Benchmarking

### Test Organization
```
tests/
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── performance/             # Performance tests
└── fixtures/               # Test data
```

## 📊 Monitoring and Logging

### Logging Strategy

#### Structured Logging
- JSON format for machine parsing
- Consistent field naming
- Correlation IDs for request tracking
- Automatic log rotation

#### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General operational information
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical system failures

### Monitoring Metrics

#### System Metrics
- CPU and memory usage
- API request rates
- Database performance
- Processing throughput

#### Business Metrics
- Concept discovery rate
- Q&A generation quality
- User engagement (if applicable)
- System uptime

## 🔒 Security Considerations

### API Key Management
- Environment-based configuration
- Secure storage practices
- Rotation strategies
- Access logging

### Data Protection
- Input validation
- Output sanitization
- Database encryption
- Access controls

### System Hardening
- Dependency scanning
- Security patches
- Network security
- Audit logging

## 🚀 Deployment Architecture

### Development Environment
- Local development setup
- Docker compose for services
- Hot reloading
- Debug configurations

### Production Environment
- Container orchestration
- Load balancing
- Auto-scaling
- Monitoring integration

### CI/CD Pipeline
- Automated testing
- Security scanning
- Deployment automation
- Rollback strategies

---

This structure provides a robust foundation for scalable AI-powered knowledge graph construction while maintaining flexibility for different use cases and deployment scenarios.