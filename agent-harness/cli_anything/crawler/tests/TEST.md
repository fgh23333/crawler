# Test Plan - CLI-Anything Crawler

## Unit Tests (test_core.py)

### TestFindProjectRoot
- test_finds_root_with_package_json: Verifies project root detection when package.json exists
- test_returns_none_when_not_found: Verifies None returned when no markers found

### TestCrawlStatus
- test_status_with_data: Verify crawl status returns all 9 subjects
- test_status_no_project_root: Verify error when no project root

### TestFetchQuestions
- test_invalid_subject: Verify error for unknown subject code
- test_valid_subject_structure: Verify fetch with mocked subprocess returns correct structure

### TestRewriteData
- test_rewrite_creates_classified_files: Verify rewrite creates singleChoice/rightWrong/etc files
- test_rewrite_nonexistent_input: Verify error for missing input directory

### TestClassifyQuestions
- test_classify: Verify classification of 4 question types (rightWrong, fillingBlank, singleChoice, multipleChoice)

### TestMergeData
- test_merge: Verify merge deduplicates and combines old+new data
- test_merge_missing_dir: Verify error for missing old directory

### TestTransformData
- test_transform: Verify transformation to standard QA format with options and labels

### TestDataStats
- test_stats: Verify stats collection from JSON files

### TestExtractConcepts
- test_extract: Verify concept extraction produces JSON and TXT outputs
- test_extract_missing_file: Verify error for missing input file

### TestGraphInfo
- test_graph_info: Verify graph node/edge counting
- test_graph_info_no_file: Verify error for missing graph file

### TestQAStats
- test_qa_stats: Verify QA statistics with subject/type breakdown

### TestShowConfig
- test_show_config: Verify config detection (project root, memcube, dependencies)

### TestCheckEnv
- test_check_env: Verify environment check lists Python/Node packages

### TestSession
- test_session_set_get: Verify session variable get/set
- test_session_persistence: Verify session data persists across instances
- test_session_history: Verify command history recording
- test_session_list: Verify session listing
- test_session_delete: Verify session deletion
- test_session_info: Verify session info output

### TestOutputFormatting
- test_json_mode: Verify JSON output is valid JSON
- test_human_mode: Verify human-readable output contains expected values
- test_error_mode: Verify error formatting
- test_json_error_mode: Verify error output in JSON mode

## E2E Tests (test_full_e2e.py)

### TestCLIEntry
- test_version: Verify --version outputs correct version
- test_help: Verify --help shows all command groups
- test_no_args_shows_help: Verify bare invocation shows help

### TestCrawlCommands
- test_crawl_help: Verify crawl subcommand help
- test_crawl_status_json: Verify crawl status with JSON output
- test_crawl_fetch_invalid_subject: Verify graceful error for invalid subject

### TestDataCommands
- test_data_help: Verify data subcommand help
- test_data_stats_json: Verify data stats with JSON output

### TestConceptCommands
- test_concept_help: Verify concept subcommand help
- test_concept_info_no_file: Verify graceful error for missing graph file

### TestQACommands
- test_qa_help: Verify QA subcommand help

### TestConfigCommands
- test_config_help: Verify config subcommand help
- test_config_check_env: Verify environment check runs successfully

### TestSessionCommands
- test_session_help: Verify session subcommand help
- test_session_list: Verify session list runs

### TestJSONOutput
- test_json_flag: Verify --json produces valid JSON output

### TestCLISubprocess
- test_installed_version: Verify CLI via subprocess --version
- test_installed_help: Verify CLI via subprocess --help
- test_installed_check_env: Verify CLI via subprocess check-env

### TestFullWorkflow
- test_full_workflow: End-to-end: rewrite -> transform -> extract concepts with real files

## Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.1, pytest-9.0.3, pluggy-1.6.0
rootdir: D:\Programs\node.js\crawler\agent-harness
plugins: cov-7.1.0
collected 50 items

cli_anything/crawler/tests/test_core.py .................... [30 passed]
cli_anything/crawler/tests/test_full_e2e.py ............... [20 passed]

============================= 50 passed in 9.63s ==============================
```

**Result: 50/50 passed (100%)**

## Coverage

- All core modules tested (crawl, data, concept, qa, config, session)
- Output formatting tested (JSON and human-readable modes)
- CLI entry point tested via subprocess
- Full workflow tested end-to-end
- Error handling tested for missing files/directories
