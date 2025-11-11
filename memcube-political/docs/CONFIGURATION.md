# Configuration Guide

## 📋 Overview

MemCube Political uses a flexible, hierarchical configuration system that allows customization of every aspect of the system's operation. Configuration is managed through YAML files with support for environment variables and command-line overrides.

## 🏗️ Configuration Architecture

### Configuration Hierarchy
1. **Default Values**: Built-in sensible defaults
2. **System Config**: `config/config.yaml`
3. **API Config**: `config/api_keys.yaml`
4. **Environment Variables**: Runtime overrides
5. **Command Line**: Immediate parameter changes

### Configuration Files

#### Primary Configuration: `config/config.yaml`
Main system configuration controlling models, processing parameters, and system behavior.

#### API Configuration: `config/api_keys.yaml`
Sensitive API keys and authentication credentials (never committed to version control).

#### Template Files: `*.example`
Configuration templates with documentation and default values.

## 🔑 API Configuration

### API Keys File (`config/api_keys.yaml`)

```yaml
# OpenAI Configuration (for GPT models)
openai:
  api_key: "your-openai-api-key-here"
  base_url: "https://api.openai.com/v1"  # Optional custom endpoint
  organization: "your-organization-id"   # Optional
  timeout: 60

# Google Gemini Configuration
google:
  api_key: "your-gemini-api-key-here"
  base_url: "https://generativelanguage.googleapis.com"  # Optional
  timeout: 60

# Anthropic Claude Configuration (via OpenAI-compatible endpoint)
anthropic:
  api_key: "your-claude-api-key-here"
  base_url: "https://api.anthropic.com"  # Example endpoint
  timeout: 60

# Custom OpenAI-Compatible API
custom:
  api_key: "your-custom-api-key"
  base_url: "https://your-custom-endpoint.com/v1"
  timeout: 60
  # Additional custom headers
  headers:
    "Custom-Header": "value"
```

### API Provider Setup

#### Google Gemini
1. Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add to `config/api_keys.yaml`:
```yaml
google:
  api_key: "your-gemini-api-key"
```

#### OpenAI
1. Create account at [OpenAI Platform](https://platform.openai.com)
2. Generate API key from dashboard
3. Add to `config/api_keys.yaml`:
```yaml
openai:
  api_key: "your-openai-api-key"
```

#### Custom API Providers
For any OpenAI-compatible API:
```yaml
custom:
  api_key: "your-api-key"
  base_url: "https://your-provider.com/v1"
```

## ⚙️ Main Configuration (`config/config.yaml`)

### API Model Configuration

```yaml
api:
  # Models for different processing stages
  model_thinker: "gemini-2.5-flash"        # Deep concept analysis
  model_extractor: "gemini-2.5-flash"     # Concept extraction
  model_expander: "gemini-2.5-flash"      # Graph expansion
  model_qa_generator: "gemini-2.5-flash"  # Q&A generation

  # API request parameters
  temperature: 0.7                        # Creativity (0.0-1.0)
  max_tokens: 32768                      # Maximum response tokens
  max_retries: 3                         # Retry attempts
  timeout: 60                           # Request timeout (seconds)

  # Rate limiting
  requests_per_minute: 60               # API rate limit
  tokens_per_minute: 100000             # Token rate limit

  # Proxy configuration (optional)
  proxy:
    http: "http://127.0.0.1:7890"       # HTTP proxy
    https: "http://127.0.0.1:7890"      # HTTPS proxy
```

#### Model Selection Guidelines

| Task | Recommended Models | Notes |
|------|-------------------|-------|
| Concept Analysis | `gemini-2.5-flash`, `gpt-4`, `claude-3-opus` | Requires deep reasoning |
| Concept Extraction | `gemini-2.5-flash`, `gpt-4o-mini` | Structured extraction |
| Graph Expansion | `gemini-2.5-flash`, `gpt-4o-mini` | Efficient processing |
| Q&A Generation | `gemini-2.5-flash`, `gpt-4` | High-quality output |

### Embedding Configuration

```yaml
embedding:
  # Model configuration
  model_name: "bge-m3"                  # Ollama model name
  model_type: "ollama"                  # Provider: ollama, openai, custom

  # Ollama-specific settings
  ollama_url: "http://localhost:11434"   # Ollama service URL
  timeout: 30                           # Request timeout

  # Processing parameters
  batch_size: 16                        # Embeddings per batch
  max_text_length: 512                  # Max text characters
  normalize_embeddings: true            # L2 normalization

  # Performance settings
  cache_embeddings: true                # Enable caching
  cache_dir: "./data/embedding_cache"   # Cache directory

  # Device configuration
  device: "cpu"                         # cpu, cuda, mps
  num_threads: 4                        # CPU threads
```

#### Embedding Model Options

| Model | Provider | Dimensions | Language | Best For |
|-------|----------|------------|----------|----------|
| bge-m3 | Ollama | 1024 | Multilingual | Chinese political concepts |
| text-embedding-3-large | OpenAI | 3072 | Multilingual | High-quality embeddings |
| text-embedding-3-small | OpenAI | 1536 | Multilingual | Cost-effective |
| sentence-transformers/* | HuggingFace | Varies | Multilingual | Local processing |

### Concept Expansion Configuration

```yaml
concept_expansion:
  # Similarity thresholds
  similarity_threshold: 0.80            # Min similarity for connections
  edge_similarity_threshold: 0.75       # Min similarity for edges

  # Growth control
  new_concept_rate_threshold: 0.10      # Min growth rate to continue
  new_edge_rate_threshold: 0.05         # Min edge growth rate

  # Processing parameters
  max_iterations: 10                    # Maximum expansion rounds
  min_concepts_per_iteration: 5         # Min concepts to add
  max_concepts_per_iteration: 100       # Max concepts per iteration

  # Batch processing
  batch_size: 50                        # Concepts per processing batch
  max_workers: 10                       # Parallel processing workers

  # Quality control
  enable_quality_filter: true           # Enable quality filtering
  min_quality_score: 0.7               # Minimum concept quality

  # Convergence criteria
  convergence_patience: 3               # Rounds without growth before stop
  min_growth_percentage: 0.01           # Min growth per iteration (1%)
```

### Concept Validation Configuration

```yaml
concept_validation:
  # Quality thresholds
  validity_threshold: 0.8               # Minimum validity score
  confidence_threshold: 0.7             # Minimum confidence score

  # Quality filters
  quality_filters:
    max_concept_length: 10             # Maximum words per concept
    min_concept_length: 1              # Minimum words per concept
    exclude_numbers: true              # Filter numeric concepts
    exclude_duplicates: true           # Remove duplicate concepts
    exclude_single_characters: true    # Remove single-character concepts

  # Validation rules
  require_political_relevance: true    # Must be politically relevant
  min_semantic_coherence: 0.6         # Minimum semantic coherence

  # Language processing
  language_detection: true             # Detect concept language
  allowed_languages: ["zh", "en"]      # Allowed language codes
  translate_to_english: false          # Auto-translate concepts
```

### Q&A Generation Configuration

```yaml
qa_generation:
  # Processing parameters
  concepts_per_batch: 20                # Concepts per processing batch
  qa_pairs_per_concept: 3               # Questions per concept
  qa_pairs_per_concept_pair: 2          # Questions per concept pair

  # Quality control
  enable_validity_check: true           # Enable quality filtering
  min_validity_threshold: 0.6           # Minimum Q&A quality
  max_question_length: 200              # Max question characters
  min_answer_length: 50                 # Min answer characters
  max_answer_length: 1000               # Max answer characters

  # Question types and distribution
  question_types:
    concept_understanding: 0.4          # 40% concept understanding
    analysis_application: 0.3           # 30% analysis application
    comparison_evaluation: 0.2          # 20% comparison evaluation
    synthesis_creation: 0.1             # 10% synthesis creation

  # Difficulty distribution
  difficulty_levels:
    easy: 0.3                           # 30% easy questions
    medium: 0.5                         # 50% medium questions
    hard: 0.2                           # 20% hard questions

  # Processing settings
  max_workers: 5                        # Parallel Q&A generation workers
  batch_processing: true                # Enable batch processing
  deduplicate_questions: true           # Remove duplicate questions

  # Quality assessment
  enable_relevance_check: true          # Check answer relevance
  enable_factuality_check: false        # Check factual accuracy (expensive)
  min_relevance_score: 0.7             # Minimum relevance threshold
```

### Database Configuration

#### Graph Database Settings

```yaml
graph_database:
  enabled: true                         # Enable graph database
  type: "neo4j"                         # Database type: neo4j, arangodb, janusgraph

  # Neo4j configuration
  neo4j:
    uri: "bolt://localhost:7687"        # Neo4j connection URI
    username: "neo4j"                   # Database username
    password: "your_password"           # Database password
    database: "neo4j"                   # Database name
    max_connection_lifetime: 3600        # Connection lifetime (seconds)
    max_connection_pool_size: 50        # Connection pool size
    connection_timeout: 30              # Connection timeout (seconds)

  # ArangoDB configuration
  arangodb:
    host: "localhost"
    port: 8529
    username: "root"
    password: "your_password"
    database: "memcube_political"

  # JanusGraph configuration
  janusgraph:
    host: "localhost"
    port: 8182
    graph_name: "political_concepts"

  # General settings
  options:
    batch_size: 100                     # Batch write size
    auto_create_indexes: true           # Automatically create indexes
    retry_attempts: 3                   # Retry failed operations
    connection_pool_size: 10            # Connection pool size
```

#### Vector Database Settings

```yaml
vector_database:
  enabled: true                         # Enable vector database
  type: "qdrant"                        # Database type: qdrant, chroma, faiss, milvus

  # Qdrant configuration
  qdrant:
    host: "localhost"
    port: 6333
    collection_name: "political_concepts"
    vector_size: 1024                   # Vector dimensions
    distance: "Cosine"                  # Distance metric: Cosine, Euclidean, Dot
    api_key: null                       # API key if required

  # ChromaDB configuration
  chroma:
    path: "./data/vector_db"            # Database storage path
    collection_name: "political_concepts"
    persist_directory: "./data/vector_db"

  # FAISS configuration (in-memory)
  faiss:
    index_type: "IVF_PQ"                # Index type: IVF_PQ, HNSW, Flat
    dimension: 1024                     # Vector dimensions
    index_path: "./data/faiss_index"
    save_interval: 100                  # Save interval

  # Milvus configuration
  milvus:
    host: "localhost"
    port: 19530
    collection_name: "political_concepts"
    vector_size: 1024
    index_type: "IVF_FLAT"

  # General settings
  options:
    batch_size: 50                      # Batch insert size
    search_top_k: 10                    # Search results count
    similarity_threshold: 0.7           # Similarity threshold
    auto_index: true                    # Auto-create indexes
    persistence: true                   # Enable persistence
```

### Data Paths Configuration

```yaml
paths:
  # Input data files
  seed_concepts: "data/seed_concepts.txt"
  seed_concepts_json: "data/seed_concepts.json"
  existing_qa_data: "data/transformed_political_data.json"

  # Output directories
  concept_graph_dir: "data/concept_graph"
  knowledge_base: "data/political_theory_knowledge_base.yaml"
  results_dir: "results"
  logs_dir: "logs"

  # Processing directories
  temp_dir: "temp"
  cache_dir: "cache"
  backup_dir: "backup"
```

### Logging Configuration

```yaml
logging:
  # Log level
  level: "INFO"                         # DEBUG, INFO, WARNING, ERROR, CRITICAL

  # Log format
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"

  # Log rotation
  rotation: "1 day"                     # Rotation frequency
  retention: "30 days"                  # Retention period

  # Log files
  file_name: "memcube_{time:YYYY-MM-DD}.log"
  log_dir: "logs"

  # Console output
  console_output: true                  # Enable console logging
  console_level: "INFO"                 # Console log level

  # Structured logging
  structured: false                     # JSON format logging
  include_metadata: true                # Include request metadata
```

### Performance Configuration

```yaml
performance:
  # Memory management
  max_memory_usage: "8GB"               # Maximum memory usage
  enable_memory_profiling: false        # Enable memory profiling
  garbage_collection_interval: 100      # GC interval

  # Processing optimization
  enable_parallel_processing: true      # Enable parallel processing
  cpu_count: null                       # CPU count (null = auto-detect)

  # Caching
  enable_api_response_cache: true       # Cache API responses
  cache_ttl: 3600                       # Cache TTL (seconds)
  max_cache_size: "1GB"                 # Maximum cache size

  # Monitoring
  enable_performance_monitoring: true   # Enable performance metrics
  metrics_interval: 60                  # Metrics collection interval

  # Rate limiting
  enable_rate_limiting: true            # Enable API rate limiting
  rate_limit_strategy: "adaptive"       # Rate limiting strategy
```

## 🔧 Environment Variables

### Supported Environment Variables

```bash
# API Keys (alternatives to config files)
export MEMCUBE_OPENAI_API_KEY="your-openai-key"
export MEMCUBE_GOOGLE_API_KEY="your-google-key"
export MEMCUBE_CUSTOM_API_KEY="your-custom-key"

# Configuration overrides
export MEMCUBE_CONFIG_PATH="/path/to/config.yaml"
export MEMCUBE_LOG_LEVEL="DEBUG"
export MEMCUBE_MAX_WORKERS="20"

# Database connections
export MEMCUBE_NEO4J_URI="bolt://localhost:7687"
export MEMCUBE_NEO4J_PASSWORD="your-password"
export MEMCUBE_QDRANT_HOST="localhost"

# Service endpoints
export MEMCUBE_OLLAMA_URL="http://localhost:11434"
export MEMCUBE_API_BASE_URL="https://custom-endpoint.com"

# Performance tuning
export MEMCUBE_BATCH_SIZE="100"
export MEMCUBE_SIMILARITY_THRESHOLD="0.85"
export MEMCUBE_MAX_ITERATIONS="15"
```

### Using Environment Variables

#### In Shell
```bash
# Set environment variables
export MEMCUBE_LOG_LEVEL="DEBUG"
export MEMCUBE_MAX_WORKERS="20"

# Run the system
python main.py --stage all
```

#### In Docker
```bash
docker run -e MEMCUBE_LOG_LEVEL=DEBUG \
           -e MEMCUBE_MAX_WORKERS=20 \
           memcube-political
```

#### In .env File
```bash
# Create .env file
cat > .env <<EOF
MEMCUBE_LOG_LEVEL=DEBUG
MEMCUBE_MAX_WORKERS=20
MEMCUBE_GOOGLE_API_KEY=your-google-key
EOF

# Load and run
export $(cat .env | xargs)
python main.py --stage all
```

## 🎛️ Command Line Configuration

### Command Line Overrides

```bash
# Override configuration values
python main.py --stage all \
               --config custom_config.yaml \
               --log-level DEBUG \
               --max-workers 20 \
               --similarity-threshold 0.85

# Quick configuration for different scenarios
python main.py --quick-start --high-performance
python main.py --quick-start --low-resource
python main.py --quick-start --development
```

### Available Command Line Options

```bash
# Configuration options
--config PATH              Custom configuration file
--log-level LEVEL          Override log level
--max-workers N            Override worker count
--batch-size N             Override batch size
--similarity-threshold N   Override similarity threshold

# Preset configurations
--high-performance         Optimized for speed
--low-resource             Optimized for minimal resources
--development              Development-friendly settings
--production               Production-optimized settings
```

## 📊 Configuration Profiles

### Development Profile

```yaml
# config/development.yaml
api:
  model_thinker: "gemini-2.5-flash"
  temperature: 0.8

concept_expansion:
  max_iterations: 3
  batch_size: 10
  max_workers: 2

qa_generation:
  concepts_per_batch: 5
  max_workers: 1

logging:
  level: "DEBUG"
  console_output: true
```

### Production Profile

```yaml
# config/production.yaml
api:
  model_thinker: "gemini-2.5-flash"
  temperature: 0.7

concept_expansion:
  max_iterations: 10
  batch_size: 100
  max_workers: 20

qa_generation:
  concepts_per_batch: 50
  max_workers: 10

logging:
  level: "INFO"
  structured: true
  console_output: false

performance:
  enable_performance_monitoring: true
  enable_caching: true
```

### High-Performance Profile

```yaml
# config/high_performance.yaml
concept_expansion:
  similarity_threshold: 0.75      # Lower threshold for more connections
  batch_size: 200                 # Large batches
  max_workers: 50                 # High parallelism

qa_generation:
  concepts_per_batch: 100         # Large batches
  max_workers: 20                 # High parallelism
  enable_validity_check: false    # Skip quality checks for speed

embedding:
  batch_size: 64                  # Large embedding batches
  cache_embeddings: true          # Enable caching

performance:
  enable_parallel_processing: true
  enable_caching: true
  max_cache_size: "4GB"
```

### Low-Resource Profile

```yaml
# config/low_resource.yaml
concept_expansion:
  similarity_threshold: 0.90      # Higher threshold for quality
  batch_size: 5                   # Small batches
  max_workers: 1                  # Single-threaded
  max_iterations: 5               # Fewer iterations

qa_generation:
  concepts_per_batch: 2           # Small batches
  max_workers: 1                  # Single-threaded
  qa_pairs_per_concept: 1         # Fewer Q&A pairs

embedding:
  batch_size: 4                   # Small embedding batches
  device: "cpu"                   # Force CPU

logging:
  level: "WARNING"                # Minimal logging
  console_output: false           # No console output
```

## 🧪 Configuration Validation

### Validate Configuration

```bash
# Check configuration syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Run comprehensive validation
python main.py --validate-config

# Test specific configuration file
python main.py --config custom_config.yaml --validate-config
```

### Common Configuration Issues

#### API Configuration Problems
```yaml
# ❌ Incorrect
api:
  model_thinker: "invalid-model"    # Model doesn't exist

# ✅ Correct
api:
  model_thinker: "gemini-2.5-flash"  # Valid model name
```

#### Threshold Configuration
```yaml
# ❌ Invalid range
concept_expansion:
  similarity_threshold: 1.5        # > 1.0, invalid

# ✅ Valid range
concept_expansion:
  similarity_threshold: 0.80        # 0.0-1.0, valid
```

#### Database Configuration
```yaml
# ❌ Missing required fields
neo4j:
  uri: "bolt://localhost:7687"
  # Missing username and password

# ✅ Complete configuration
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "your-password"
```

## 🔄 Configuration Updates

### Hot Configuration Updates

Some configuration parameters can be updated at runtime:

```bash
# Update log level
python main.py --update-config logging.level=DEBUG

# Update worker count
python main.py --update-config performance.max_workers=15

# Update similarity threshold
python main.py --update-config concept_expansion.similarity_threshold=0.85
```

### Configuration Reloading

```bash
# Reload configuration from file
python main.py --reload-config

# Reload specific section
python main.py --reload-config api
python main.py --reload-config concept_expansion
```

### Configuration Backup

```bash
# Backup current configuration
cp config/config.yaml backup/config_$(date +%Y%m%d_%H%M%S).yaml

# Export configuration summary
python main.py --export-config-summary > config_summary.txt
```

## 🛠️ Advanced Configuration

### Custom Model Configuration

```yaml
# Add custom model configuration
custom_models:
  political_analyzer:
    provider: "custom"
    model_name: "political-theory-v1"
    api_base: "https://your-api.com/v1"
    api_key_env: "CUSTOM_API_KEY"
    max_tokens: 4096
    temperature: 0.5
```

### Pipeline Customization

```yaml
# Custom processing pipeline
pipeline:
  stages:
    - name: "preprocessing"
      enabled: true
      config:
        normalize_text: true
        remove_duplicates: true

    - name: "concept_analysis"
      enabled: true
      config:
        deep_analysis: true
        extract_relationships: true

    - name: "quality_control"
      enabled: true
      config:
        strict_validation: true
```

### Feature Flags

```yaml
# Enable/disable experimental features
features:
  experimental_embeddings: false
  advanced_quality_filters: true
  real_time_monitoring: true
  auto_optimization: false
```

---

This comprehensive configuration system allows fine-tuned control over all aspects of MemCube Political's operation, from model selection to performance optimization.