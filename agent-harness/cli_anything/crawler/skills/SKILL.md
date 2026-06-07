---
name: cli-anything-crawler
version: 1.0.0
description: CLI harness for Political Theory Question Bank Crawler and Knowledge Graph System
entry_point: cli_anything.crawler.crawler_cli:cli
commands:
  - name: crawl fetch
    group: crawl
    description: Fetch questions from the exam system
    options:
      - name: --subject
        shorthand: -s
        description: Subject code(s) to fetch
        required: false
      - name: --all
        description: Fetch all 9 subjects
        required: false
      - name: --output
        shorthand: -o
        description: Output directory
        required: false
      - name: --iterations
        shorthand: -n
        description: Number of fetch iterations per subject (default 1000)
        required: false
    examples:
      - cli-anything-crawler crawl fetch --subject Marx
      - cli-anything-crawler crawl fetch --all --iterations 500
  - name: crawl status
    group: crawl
    description: Show crawl status for all subjects
    options:
      - name: --dir
        description: Data directory to check
        required: false
    examples:
      - cli-anything-crawler crawl status
  - name: data rewrite
    group: data
    description: Rewrite raw solved data into classified format
    options:
      - name: --input
        shorthand: -i
        description: Input directory containing solved JSON files
        required: false
      - name: --output
        shorthand: -o
        description: Output directory for classified files
        required: false
    examples:
      - cli-anything-crawler data rewrite
      - cli-anything-crawler data rewrite --input 2025-05-27/solved --output 2025-05-27/rewrite
  - name: data classify
    group: data
    description: Classify questions by type
    options:
      - name: --input
        shorthand: -i
        description: Directory containing JSON files
        required: false
    examples:
      - cli-anything-crawler data classify --input merge/
  - name: data merge
    group: data
    description: Merge old and new question data with deduplication
    options:
      - name: --old
        description: Old data directory
        required: false
      - name: --new
        description: New data directory
        required: false
      - name: --output
        shorthand: -o
        description: Output directory
        required: false
    examples:
      - cli-anything-crawler data merge --old new/rewrite --new 2025-05-27/rewrite --output merge/
  - name: data transform
    group: data
    description: Transform classified data into standard QA format
    options:
      - name: --input
        shorthand: -i
        description: Input directory (rewrite/)
        required: false
      - name: --output
        shorthand: -o
        description: Output JSON file
        required: false
    examples:
      - cli-anything-crawler data transform
  - name: data stats
    group: data
    description: Show statistics for question data
    options:
      - name: --dir
        description: Directory to scan
        required: false
    examples:
      - cli-anything-crawler data stats --dir merge/
  - name: concept extract
    group: concept
    description: Extract seed concepts from QA data
    options:
      - name: --input
        shorthand: -i
        description: Input QA JSON file
        required: false
      - name: --output
        shorthand: -o
        description: Output directory
        required: false
    examples:
      - cli-anything-crawler concept extract
  - name: concept expand
    group: concept
    description: Run concept graph expansion via MemCube Political
    options:
      - name: --config
        shorthand: -c
        description: Config file path
        required: false
      - name: --stage
        description: Expansion stage (concept-analysis, concept-extraction, concept-expansion)
        required: false
    examples:
      - cli-anything-crawler concept expand --stage concept-expansion
  - name: concept info
    group: concept
    description: Show concept graph information
    options:
      - name: --file
        shorthand: -f
        description: Concept graph JSON file
        required: false
    examples:
      - cli-anything-crawler concept info
  - name: qa generate
    group: qa
    description: Generate QA pairs from concept graph
    options:
      - name: --graph
        shorthand: -g
        description: Concept graph file
        required: false
      - name: --config
        shorthand: -c
        description: Config file path
        required: false
    examples:
      - cli-anything-crawler qa generate
  - name: qa pipeline
    group: qa
    description: Run full pipeline (concept expansion + QA generation)
    options:
      - name: --config
        shorthand: -c
        description: Config file path
        required: false
    examples:
      - cli-anything-crawler qa pipeline
  - name: qa stats
    group: qa
    description: Show QA statistics
    options:
      - name: --file
        shorthand: -f
        description: QA JSON file
        required: false
    examples:
      - cli-anything-crawler qa stats
  - name: config show
    group: config
    description: Show current configuration and dependencies
    examples:
      - cli-anything-crawler config show
  - name: config check-env
    group: config
    description: Check environment dependencies
    examples:
      - cli-anything-crawler config check-env
  - name: config test-api
    group: config
    description: Test API configuration
    examples:
      - cli-anything-crawler config test-api
  - name: session list
    group: session
    description: List all persistent sessions
    examples:
      - cli-anything-crawler session list
  - name: repl
    group: repl
    description: Start interactive REPL mode
    options:
      - name: --session
        shorthand: -s
        description: Session name
        required: false
    examples:
      - cli-anything-crawler repl
      - cli-anything-crawler repl --session my-session
---

# CLI-Anything Crawler

A stateful CLI harness for the Political Theory Question Bank Crawler and Knowledge Graph System.

## Command Groups

### crawl - Data Fetching
Fetch political theory exam questions from 9 subjects via the exam system API.

### data - Data Processing
Process and transform question data through the pipeline:
rewrite → classify → merge → transform → stats

### concept - Concept Graph Management
Manage concept graphs using MemCube Political:
extract seed concepts → expand graph → view graph info

### qa - QA Generation
Generate and manage QA pairs from concept graphs.

### config - Configuration
Manage configuration and check environment dependencies.

### session - Session Management
Manage persistent sessions for stateful operations.

## Agent Usage Guidelines

1. **Always check environment first**: Run `config check-env` to verify dependencies
2. **Typical pipeline**: `crawl fetch → data rewrite → data transform → concept extract → concept expand → qa generate`
3. **Use --json flag** for machine-readable output when parsing results programmatically
4. **Session persistence**: Use `--session <name>` or `repl --session <name>` to maintain state across invocations

## Supported Subjects

XiIntro, Marx, MaoIntro, Political, CMH, NCH, SDH, ORH, CCPH

## Question Types

singleChoice, multipleChoice, rightWrong, fillingBlank, subject
