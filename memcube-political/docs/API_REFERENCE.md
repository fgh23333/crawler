# API Reference

## 📋 Overview

MemCube Political provides both Python APIs and command-line interfaces for programmatic access to all system functionality. This reference documents all available APIs, parameters, and usage examples.

## 🐍 Python APIs

### Core Modules

#### ConceptAnalyzer

Analyzes political theory concepts using advanced language models.

```python
from src.concept_analyzer import ConceptAnalyzer

# Initialize analyzer
analyzer = ConceptAnalyzer(config_path="config/config.yaml")

# Single concept analysis
result = analyzer.analyze_concept("democracy")

# Batch concept analysis
concepts = ["democracy", "constitutional law", "civil rights"]
results = analyzer.analyze_concepts_batch(concepts)
```

**Parameters:**
- `config_path` (str): Path to configuration file
- `api_client` (APIClient): Custom API client instance

**Returns:**
- `ConceptAnalysisResult`: Analysis results with extracted concepts and relationships

**Methods:**

```python
def analyze_concept(concept: str) -> Dict[str, Any]:
    """Analyze a single concept in depth."""

def analyze_concepts_batch(concepts: List[str]) -> List[Dict[str, Any]]:
    """Analyze multiple concepts in batch."""

def validate_analysis_result(result: Dict[str, Any]) -> bool:
    """Validate analysis result quality."""
```

#### ConceptExtractor

Extracts and cleans concepts from analysis text.

```python
from src.concept_extractor import ConceptExtractor

# Initialize extractor
extractor = ConceptExtractor(config_path="config/config.yaml")

# Extract concepts from text
text = "Democracy is a system of government where citizens exercise power..."
concepts = extractor.extract_concepts(text)

# Filter and validate concepts
filtered_concepts = extractor.filter_concepts(concepts)
```

**Parameters:**
- `config_path` (str): Path to configuration file
- `quality_threshold` (float): Minimum quality score for concepts

**Methods:**

```python
def extract_concepts(text: str) -> List[str]:
    """Extract concepts from analysis text."""

def filter_concepts(concepts: List[str]) -> List[str]:
    """Filter concepts based on quality criteria."""

def normalize_concept(concept: str) -> str:
    """Normalize concept format."""
```

#### ConceptGraph

Builds and expands knowledge graphs from concepts.

```python
from src.concept_graph import ConceptGraph

# Initialize with seed concepts
seed_concepts = ["democracy", "constitutional law", "civil rights"]
graph = ConceptGraph(seed_concepts, config_path="config/config.yaml")

# Run full expansion
result = graph.run_full_expansion()

# Manual expansion step
new_concepts = graph.expand_concepts_step()
graph.add_concepts(new_concepts)

# Save/load graph
graph.save_graph("path/to/graph.json")
graph.load_graph("path/to/graph.json")
```

**Parameters:**
- `seed_concepts` (List[str]): Initial concepts for graph
- `config_path` (str): Path to configuration file
- `embedding_client` (EmbeddingClient): Custom embedding client

**Methods:**

```python
def run_full_expansion(self) -> Dict[str, Any]:
    """Run complete graph expansion process."""

def expand_concepts_step(self) -> List[str]:
    """Perform one expansion iteration."""

def check_convergence(self) -> bool:
    """Check if graph expansion has converged."""

def get_graph_statistics(self) -> Dict[str, Any]:
    """Get graph statistics and metrics."""

def save_graph(self, file_path: str) -> bool:
    """Save graph to file."""

def load_graph(self, file_path: str) -> bool:
    """Load graph from file."""
```

#### QAGenerator

Generates question-answer pairs from concepts and concept graphs.

```python
from src.qa_generator import QAGenerator

# Initialize generator
generator = QAGenerator(config_path="config/config.yaml")

# Load concept graph
generator.load_concept_graph("path/to/graph.json")

# Generate Q&A for single concepts
qa_pairs = generator.generate_single_concept_qa(["democracy", "freedom"])

# Generate Q&A for concept pairs
qa_pairs = generator.generate_concept_pair_qa([("democracy", "freedom")])

# Run full generation pipeline
result = generator.run_full_qa_generation("path/to/graph.json")
```

**Parameters:**
- `config_path` (str): Path to configuration file
- `api_client` (APIClient): Custom API client

**Methods:**

```python
def generate_single_concept_qa(self, concepts: List[str]) -> List[Dict[str, Any]]:
    """Generate Q&A pairs for individual concepts."""

def generate_concept_pair_qa(self, concept_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Generate Q&A pairs for concept relationships."""

def filter_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter and validate Q&A pairs."""

def deduplicate_qa_pairs(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate Q&A pairs."""

def run_full_qa_generation(self, graph_file: str) -> Dict[str, Any]:
    """Run complete Q&A generation process."""
```

#### Evaluation

Comprehensive quality assessment of generated content.

```python
from src.evaluation import ComprehensiveEvaluator

# Initialize evaluator
evaluator = ComprehensiveEvaluator(config_path="config/config.yaml")

# Evaluate concept graph
graph_report = evaluator.evaluate_graph("path/to/graph.json")

# Evaluate Q&A dataset
qa_report = evaluator.evaluate_qa_dataset("path/to/qa_dataset.json")

# Full system evaluation
full_report = evaluator.evaluate_full_system(
    graph_file="path/to/graph.json",
    qa_file="path/to/qa_dataset.json"
)
```

**Parameters:**
- `config_path` (str): Path to configuration file

**Methods:**

```python
def evaluate_graph(self, graph_file: str) -> Dict[str, Any]:
    """Evaluate concept graph quality."""

def evaluate_qa_dataset(self, qa_file: str) -> Dict[str, Any]:
    """Evaluate Q&A dataset quality."""

def evaluate_full_system(self, graph_file: str, qa_file: str) -> Dict[str, Any]:
    """Evaluate entire system output."""

def generate_report(self, results: Dict[str, Any]) -> str:
    """Generate human-readable evaluation report."""
```

### Database Clients

#### APIClient

Unified client for various LLM APIs.

```python
from src.api_client import APIClient

# Initialize client
client = APIClient(config_path="config/api_keys.yaml")

# Make API calls
response = client.completion(
    prompt="Analyze the concept of democracy...",
    model="gemini-2.5-flash",
    temperature=0.7,
    max_tokens=1000
)

# Batch processing
prompts = ["Analyze democracy", "Explain freedom", "Define justice"]
responses = client.batch_completion(prompts, model="gemini-2.5-flash")
```

**Parameters:**
- `config_path` (str): Path to API configuration file
- `provider` (str): API provider (openai, google, custom)

**Methods:**

```python
def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
    """Generate completion for single prompt."""

def batch_completion(self, prompts: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Generate completions for multiple prompts."""

def get_models(self) -> List[str]:
    """Get available models for provider."""

def test_connection(self) -> bool:
    """Test API connection."""
```

#### EmbeddingClient

Client for text embedding generation.

```python
from src.embedding_client import EmbeddingClient

# Initialize client
client = EmbeddingClient(config_path="config/config.yaml")

# Generate embeddings
texts = ["democracy", "freedom", "justice"]
embeddings = client.get_embeddings(texts)

# Batch processing
large_text_list = ["concept1", "concept2", ...]  # Large list
embeddings = client.get_embeddings_batch(large_text_list, batch_size=32)

# Similarity search
query_text = "government by the people"
similar_texts = client.similarity_search(query_text, top_k=5)
```

**Parameters:**
- `config_path` (str): Path to configuration file
- `model_name` (str): Embedding model name

**Methods:**

```python
def get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for texts."""

def get_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Generate embeddings in batches."""

def similarity_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find most similar texts."""

def compute_similarity(self, text1: str, text2: str) -> float:
    """Compute similarity between two texts."""
```

#### GraphDatabaseClient

Abstract client for graph databases.

```python
from src.graph_database_client import get_graph_database_client

# Get configured client
client = get_graph_database_client("config/config.yaml")

# Create concepts and relationships
client.create_concept_node("democracy", {"type": "political_system"})
client.create_concept_node("voting", {"type": "process"})

client.create_relationship("democracy", "HAS_PROCESS", "voting")

# Query the graph
results = client.query_concepts("democracy")
neighbors = client.get_neighbors("democracy")

# Batch operations
concepts = [
    ("democracy", {"type": "system"}),
    ("freedom", {"type": "value"}),
    ("justice", {"type": "principle"})
]
client.create_concepts_batch(concepts)
```

**Parameters:**
- `config_path` (str): Path to configuration file

**Methods:**

```python
def create_concept_node(self, concept: str, attributes: Dict[str, Any]) -> bool:
    """Create a concept node in the graph."""

def create_relationship(self, from_concept: str, relation: str, to_concept: str) -> bool:
    """Create relationship between concepts."""

def query_concepts(self, query: str) -> List[Dict[str, Any]]:
    """Query concepts in the graph."""

def get_neighbors(self, concept: str) -> List[Dict[str, Any]]:
    """Get neighboring concepts."""

def create_concepts_batch(self, concepts: List[Tuple[str, Dict[str, Any]]]) -> bool:
    """Create multiple concept nodes."""
```

#### VectorDatabaseClient

Abstract client for vector databases.

```python
from src.vector_database_client import get_vector_database_client

# Get configured client
client = get_vector_database_client("config/config.yaml")

# Add embeddings
concepts = ["democracy", "freedom", "justice"]
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]

client.add_embeddings(concepts, embeddings)

# Search similar concepts
query_embedding = [0.1, 0.2, ...]
similar_concepts = client.search_similar(query_embedding, top_k=5)

# Batch operations
concept_embeddings = {
    "democracy": [0.1, 0.2, ...],
    "freedom": [0.3, 0.4, ...],
    ...
}
client.add_embeddings_batch(concept_embeddings)
```

**Parameters:**
- `config_path` (str): Path to configuration file

**Methods:**

```python
def add_embeddings(self, concepts: List[str], embeddings: List[List[float]]) -> bool:
    """Add embeddings to database."""

def search_similar(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar concepts."""

def add_embeddings_batch(self, concept_embeddings: Dict[str, List[float]]) -> bool:
    """Add multiple embeddings."""

def get_embedding(self, concept: str) -> Optional[List[float]]:
    """Get embedding for specific concept."""

def delete_embeddings(self, concepts: List[str]) -> bool:
    """Delete embeddings from database."""
```

## 🔧 Utility Functions

### Configuration Loading

```python
from src.utils import load_config, validate_config

# Load configuration
config = load_config("config/config.yaml")

# Validate configuration
is_valid, errors = validate_config(config)

# Get specific configuration section
api_config = config.get("api", {})
```

### Environment Setup

```python
from src.utils import setup_logging, check_environment

# Setup logging
logger = setup_logging(
    level="INFO",
    format="{time} | {level} | {message}",
    file_path="logs/system.log"
)

# Check environment
env_status = check_environment()
if not env_status["valid"]:
    print("Environment setup incomplete:", env_status["errors"])
```

### Data Processing

```python
from src.utils import process_concepts, validate_qa_pairs

# Process concept list
processed_concepts = process_concepts(
    concepts=[" raw concept ", "concept2", "CONCEPT3"],
    normalize=True,
    remove_duplicates=True,
    min_length=1
)

# Validate Q&A pairs
valid_qa = validate_qa_pairs(
    qa_pairs=[{
        "question": "What is democracy?",
        "answer": "Democracy is a system...",
        "concept": "democracy"
    }],
    min_question_length=10,
    min_answer_length=50
)
```

## 🖥️ Command Line Interface

### Main Entry Point

```bash
# Usage: python main.py [OPTIONS] COMMAND [ARGS]

# Available commands
python main.py --help

# System operations
python main.py --check-env                    # Check environment setup
python main.py --test-api                     # Test API connections
python main.py --validate-config              # Validate configuration

# Processing stages
python main.py --stage concept-expansion      # Run concept graph expansion
python main.py --stage qa-generation          # Run Q&A generation
python main.py --stage all                     # Run complete pipeline

# Quick start options
python main.py --quick-start                  # Run with default settings
python main.py --quick-start-db               # Run with database mode
```

### Command Line Options

```bash
# Configuration options
--config PATH               Custom configuration file path
--log-level LEVEL           Override log level (DEBUG, INFO, WARNING, ERROR)
--output-dir PATH          Custom output directory
--temp-dir PATH           Custom temporary directory

# Processing options
--stage STAGE              Processing stage
--concepts-file PATH       Custom seed concepts file
--graph-file PATH          Input graph file for Q&A generation
--max-iterations N         Override max iterations
--similarity-threshold N   Override similarity threshold
--batch-size N            Override batch size
--max-workers N           Override worker count

# Quality options
--quality-threshold N     Set quality threshold
--enable-quality-check    Enable quality validation
--disable-quality-check   Disable quality validation

# Performance options
--high-performance        Use high-performance settings
--low-resource           Use low-resource settings
--memory-limit N         Set memory limit (GB)
--cpu-count N            Set CPU count

# Output options
--verbose                Enable verbose output
--quiet                  Minimal output
--dry-run               Show what would be done
--force                 Overwrite existing files

# Development options
--debug                 Enable debug mode
--profile               Enable profiling
--test-mode            Run in test mode with small dataset
```

### Script Utilities

#### Environment Check (`scripts/check_env.py`)

```bash
# Run environment validation
python scripts/check_env.py

# With specific requirements file
python scripts/check_env.py --requirements requirements.txt

# Check specific components
python scripts/check_env.py --check-api --check-ollama --check-database
```

#### Quick Start (`scripts/quick_start.py`)

```bash
# Quick start with memory mode
python scripts/quick_start.py

# Quick start with database mode
python scripts/quick_start_database.py

# Custom quick start with parameters
python scripts/quick_start.py \
    --concepts data/my_concepts.txt \
    --output results/my_run \
    --config custom_config.yaml
```

#### API Testing (`scripts/test_api_simple.py`)

```bash
# Test all configured APIs
python scripts/test_api_simple.py

# Test specific API provider
python scripts/test_api_simple.py --provider openai
python scripts/test_api_simple.py --provider google

# Test with custom configuration
python scripts/test_api_simple.py --config custom_api_config.yaml
```

## 📊 Data Structures

### Concept Graph Format

```json
{
  "graph": {
    "democracy": {
      "attributes": {
        "type": "political_system",
        "quality_score": 0.95,
        "source": "seed_concept"
      },
      "relationships": {
        "HAS_PRINCIPLE": ["freedom", "equality"],
        "HAS_PROCESS": ["voting", "election"],
        "SIMILAR_TO": ["republic", "constitutional_government"]
      }
    }
  },
  "embeddings": {
    "democracy": [0.1, 0.2, 0.3, ...]
  },
  "metadata": {
    "total_concepts": 5000,
    "total_relationships": 15000,
    "expansion_iterations": 8,
    "similarity_threshold": 0.80,
    "created_at": "2025-11-11T12:00:00Z"
  }
}
```

### Q&A Dataset Format

```json
{
  "metadata": {
    "total_qa_pairs": 10000,
    "concepts_covered": 3000,
    "question_types": {
      "concept_understanding": 4000,
      "analysis_application": 3000,
      "comparison_evaluation": 2000,
      "synthesis_creation": 1000
    },
    "difficulty_levels": {
      "easy": 3000,
      "medium": 5000,
      "hard": 2000
    },
    "generation_model": "gemini-2.5-flash",
    "created_at": "2025-11-11T12:00:00Z"
  },
  "qa_pairs": [
    {
      "id": "qa_001",
      "question": "What are the core principles of democracy?",
      "answer": "The core principles of democracy include popular sovereignty, political equality, freedom of speech, protection of human rights, and the rule of law. These principles ensure that power derives from the people and that individual freedoms are protected within a framework of equal participation.",
      "type": "concept_understanding",
      "difficulty": "medium",
      "concept": "democracy",
      "related_concepts": ["freedom", "equality", "voting"],
      "source": "single_concept",
      "quality_score": 0.92,
      "word_count": {
        "question": 8,
        "answer": 45
      },
      "created_at": "2025-11-11T12:00:00Z"
    }
  ]
}
```

### Evaluation Report Format

```json
{
  "metadata": {
    "evaluation_timestamp": "2025-11-11T12:00:00Z",
    "evaluator_version": "1.0.0",
    "configuration": {
      "graph_file": "data/concept_graph/final_graph.json",
      "qa_file": "results/qa_dataset.json"
    }
  },
  "graph_evaluation": {
    "structural_metrics": {
      "total_nodes": 5000,
      "total_edges": 15000,
      "average_degree": 6.0,
      "graph_density": 0.0012,
      "clustering_coefficient": 0.15,
      "connected_components": 1
    },
    "semantic_metrics": {
      "concept_diversity": 0.85,
      "embedding_quality": 0.92,
      "relationship_strength": 0.78
    },
    "coverage_metrics": {
      "domain_coverage": 0.95,
      "concept_depth": 0.88,
      "cross_domain_connections": 0.72
    },
    "overall_score": 0.87
  },
  "qa_evaluation": {
    "quantity_metrics": {
      "total_qa_pairs": 10000,
      "unique_questions": 9950,
      "unique_concepts": 3000
    },
    "quality_metrics": {
      "average_question_length": 25,
      "average_answer_length": 120,
      "format_correctness": 0.98,
      "relevance_score": 0.89
    },
    "diversity_metrics": {
      "question_type_distribution": {
        "concept_understanding": 0.4,
        "analysis_application": 0.3,
        "comparison_evaluation": 0.2,
        "synthesis_creation": 0.1
      },
      "difficulty_distribution": {
        "easy": 0.3,
        "medium": 0.5,
        "hard": 0.2
      }
    },
    "overall_score": 0.91
  },
  "recommendations": [
    "Increase concept diversity by including more political philosophy concepts",
    "Add more comparative analysis questions",
    "Consider expanding to include historical political systems",
    "Improve coverage of non-Western political theories"
  ]
}
```

## 🔌 Integration Examples

### Custom Pipeline

```python
from src.concept_analyzer import ConceptAnalyzer
from src.concept_graph import ConceptGraph
from src.qa_generator import QAGenerator

class CustomPoliticalAnalysisPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.analyzer = ConceptAnalyzer(config_path)
        self.graph_builder = ConceptGraph([], config_path)
        self.qa_generator = QAGenerator(config_path)

    def process_custom_concepts(self, concepts: List[str]):
        # Analyze concepts
        analysis_results = self.analyzer.analyze_concepts_batch(concepts)

        # Extract concepts
        all_concepts = []
        for result in analysis_results:
            extracted = self.extractor.extract_concepts(result["text"])
            all_concepts.extend(extracted)

        # Build graph
        self.graph_builder = ConceptGraph(all_concepts)
        graph_result = self.graph_builder.run_full_expansion()

        # Generate Q&A
        qa_result = self.qa_generator.run_full_qa_generation(
            graph_result["output_file"]
        )

        return {
            "graph": graph_result,
            "qa": qa_result
        }

# Usage
pipeline = CustomPoliticalAnalysisPipeline("config/config.yaml")
results = pipeline.process_custom_concepts(["democracy", "justice"])
```

### Database Integration

```python
from src.graph_database_client import get_graph_database_client
from src.vector_database_client import get_vector_database_client

class DatabaseManager:
    def __init__(self, config_path: str):
        self.graph_db = get_graph_database_client(config_path)
        self.vector_db = get_vector_database_client(config_path)

    def store_concept_graph(self, graph_data: Dict[str, Any]):
        # Store concepts and relationships in graph database
        for concept, data in graph_data["graph"].items():
            self.graph_db.create_concept_node(concept, data["attributes"])

            # Store relationships
            for rel_type, targets in data["relationships"].items():
                for target in targets:
                    self.graph_db.create_relationship(concept, rel_type, target)

        # Store embeddings in vector database
        concepts = list(graph_data["embeddings"].keys())
        embeddings = list(graph_data["embeddings"].values())
        self.vector_db.add_embeddings(concepts, embeddings)

    def search_concepts(self, query: str, top_k: int = 10):
        # Get embedding for query
        embedding_client = EmbeddingClient()
        query_embedding = embedding_client.get_embeddings([query])[0]

        # Search vector database
        similar_concepts = self.vector_db.search_similar(query_embedding, top_k)

        # Get detailed information from graph database
        results = []
        for concept, score in similar_concepts:
            concept_data = self.graph_db.query_concepts(concept)
            results.append({
                "concept": concept,
                "similarity": score,
                "data": concept_data
            })

        return results

# Usage
db_manager = DatabaseManager("config/config.yaml")
db_manager.store_concept_graph(graph_data)
search_results = db_manager.search_concepts("systems of government")
```

## 🚨 Error Handling

### Common Exceptions

```python
try:
    analyzer = ConceptAnalyzer("config/config.yaml")
    result = analyzer.analyze_concept("democracy")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except APIConnectionError as e:
    print(f"API connection failed: {e}")
except ConceptAnalysisError as e:
    print(f"Concept analysis failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Error Recovery

```python
from src.utils import retry_with_backoff

@retry_with_backoff(max_attempts=3, backoff_factor=2)
def robust_api_call(prompt: str):
    client = APIClient()
    return client.completion(prompt=prompt, model="gemini-2.5-flash")
```

### Validation

```python
from src.utils import validate_concepts, validate_qa_format

# Validate concepts
concepts = ["democracy", "", "freedom123", "justice"]
valid_concepts, errors = validate_concepts(concepts)
print(f"Valid concepts: {valid_concepts}")
print(f"Errors: {errors}")

# Validate Q&A format
qa_pair = {
    "question": "What is democracy?",
    "answer": "Too short"
}
is_valid, issues = validate_qa_format(qa_pair)
if not is_valid:
    print(f"Q&A validation issues: {issues}")
```

---

This comprehensive API reference provides all the information needed to integrate and extend MemCube Political programmatically.