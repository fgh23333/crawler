# CLI-Anything Crawler

A stateful CLI harness for the Political Theory Question Bank Crawler and Knowledge Graph System.

## Installation

```bash
# From the agent-harness directory
pip install -e .
```

## Usage

### One-shot Commands

```bash
# Check environment
cli-anything-crawler config check-env

# Fetch questions
cli-anything-crawler crawl fetch --subject Marx
cli-anything-crawler crawl fetch --all

# Process data pipeline
cli-anything-crawler data rewrite
cli-anything-crawler data transform
cli-anything-crawler data stats

# Concept graph operations
cli-anything-crawler concept extract
cli-anything-crawler concept expand --stage concept-expansion
cli-anything-crawler concept info

# QA generation
cli-anything-crawler qa generate
cli-anything-crawler qa pipeline

# JSON output for agent consumption
cli-anything-crawler --json crawl status
cli-anything-crawler --json data stats --dir merge/
```

### REPL Mode

```bash
cli-anything-crawler repl
cli-anything-crawler repl --session my-session
```

### Session Persistence

```bash
# Use session across commands
cli-anything-crawler --session my-work crawl fetch --subject Marx
cli-anything-crawler --session my-work data rewrite
cli-anything-crawler session list
```

## Command Groups

| Group     | Description                                |
|-----------|--------------------------------------------|
| `crawl`   | Fetch questions from exam system            |
| `data`    | Process, classify, merge, transform data   |
| `concept` | Manage concept graphs and seed concepts    |
| `qa`      | Generate and manage QA pairs               |
| `config`  | Configuration and environment management   |
| `session` | Persistent session management              |

## Supported Subjects

XiIntro, Marx, MaoIntro, Political, CMH, NCH, SDH, ORH, CCPH

## Testing

```bash
cd agent-harness
PYTHONPATH=. python -m pytest cli_anything/crawler/tests/ -v --tb=no
```
