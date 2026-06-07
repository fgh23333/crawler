#!/usr/bin/env python3
"""CLI-Anything Crawler - Political Theory Question Bank & Knowledge Graph CLI.

A stateful CLI harness for the crawler project, supporting data fetching,
processing, concept graph operations, and QA generation.
"""

import cmd
import json
import sys
from pathlib import Path

import click

from cli_anything.crawler import __version__
from cli_anything.crawler.core.crawl import fetch_questions, fetch_all, crawl_status, login, SUBJECTS
from cli_anything.crawler.core.data import (
    rewrite_data, classify_questions, merge_data, transform_data, data_stats,
)
from cli_anything.crawler.core.concept import extract_concepts, run_concept_expansion, graph_info
from cli_anything.crawler.core.qa import generate_qa, run_full_pipeline, qa_stats
from cli_anything.crawler.core.config import show_config, check_env, test_api
from cli_anything.crawler.core.session import Session
from cli_anything.crawler.utils.output import format_output

# Global state
_session: Session = None
_json_mode: bool = False


def _output(data: dict):
    """Format and print output."""
    click.echo(format_output(data, _json_mode))


def _session_record(command: str, result: dict):
    """Record command in session history."""
    global _session
    if _session:
        status = result.get("status", "unknown")
        _session.add_history(command, f"status={status}")


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="cli-anything-crawler")
@click.option("--json", "json_mode", is_flag=True, help="Output in JSON format")
@click.option("--session", "session_name", default=None, help="Session name for state persistence")
@click.pass_context
def cli(ctx, json_mode: bool, session_name: str):
    """CLI-Anything Crawler - Political Theory Question Bank & Knowledge Graph CLI.

    Manage political theory question data: fetch, process, transform, build concept
    graphs, and generate QA pairs.
    """
    global _session, _json_mode
    _json_mode = json_mode
    if session_name:
        _session = Session(name=session_name)
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── Crawl commands ──────────────────────────────────────────────────────────

@cli.group()
def crawl():
    """Fetch question data from the exam system."""
    pass


@crawl.command("login")
@click.option("--student-num", "-u", prompt=True, help="Student number")
@click.option("--password", "-p", prompt=True, hide_input=True, help="Password")
def crawl_login(student_num, password):
    """Login to the exam system (saves credentials for later use)."""
    result = login(student_num=student_num, password=password)
    _output(result)
    _session_record("crawl login", result)


@crawl.command("fetch")
@click.option("--subject", "-s", "subjects", multiple=True, help="Subject code(s) to fetch")
@click.option("--all", "fetch_all_flag", is_flag=True, help="Fetch all subjects")
@click.option("--output", "-o", "output_dir", default=None, help="Output directory")
@click.option("--iterations", "-n", default=1000, help="Number of fetch iterations per subject")
@click.option("--student-id", default=None, help="Student ID (skip login if provided)")
@click.option("--student-num", "-u", default=None, help="Student number (for login)")
@click.option("--password", "-p", default=None, help="Password (for login)")
def crawl_fetch(subjects, fetch_all_flag, output_dir, iterations, student_id, student_num, password):
    """Fetch questions from the exam system."""
    if fetch_all_flag:
        result = fetch_all(output_dir=output_dir, iterations=iterations,
                           student_num=student_num, password=password)
    elif subjects:
        results = []
        for s in subjects:
            r = fetch_questions(s, output_dir=output_dir, iterations=iterations,
                                student_id=student_id, student_num=student_num, password=password)
            results.append(r)
            click.echo(format_output(r, _json_mode))
        result = {"status": "done", "subjects_fetched": len(results)}
    else:
        click.echo("Specify --subject <code> or --all. Subjects: " + ", ".join(SUBJECTS.keys()))
        return
    _output(result)
    _session_record("crawl fetch", result)


@crawl.command("status")
@click.option("--dir", "data_dir", default=None, help="Data directory to check")
def crawl_status_cmd(data_dir):
    """Show crawl status for all subjects."""
    result = crawl_status(data_dir=data_dir)
    _output(result)
    _session_record("crawl status", result)


# ── Data commands ───────────────────────────────────────────────────────────

@cli.group()
def data():
    """Process and transform question data."""
    pass


@data.command("rewrite")
@click.option("--input", "-i", "input_dir", default=None, help="Input directory (solved/)")
@click.option("--output", "-o", "output_dir", default=None, help="Output directory (rewrite/)")
def data_rewrite(input_dir, output_dir):
    """Rewrite raw solved data into classified format."""
    result = rewrite_data(input_dir=input_dir, output_dir=output_dir)
    _output(result)
    _session_record("data rewrite", result)


@data.command("classify")
@click.option("--input", "-i", "input_dir", default=None, help="Directory containing JSON files")
def data_classify(input_dir):
    """Classify questions by type."""
    result = classify_questions(input_dir=input_dir)
    _output(result)
    _session_record("data classify", result)


@data.command("merge")
@click.option("--old", "old_dir", default=None, help="Old data directory")
@click.option("--new", "new_dir", default=None, help="New data directory")
@click.option("--output", "-o", "output_dir", default=None, help="Output directory")
def data_merge(old_dir, new_dir, output_dir):
    """Merge old and new question data."""
    result = merge_data(old_dir=old_dir, new_dir=new_dir, output_dir=output_dir)
    _output(result)
    _session_record("data merge", result)


@data.command("transform")
@click.option("--input", "-i", "input_dir", default=None, help="Input directory (rewrite/)")
@click.option("--output", "-o", "output_file", default=None, help="Output JSON file")
def data_transform(input_dir, output_file):
    """Transform classified data into standard QA format."""
    result = transform_data(input_dir=input_dir, output_file=output_file)
    _output(result)
    _session_record("data transform", result)


@data.command("stats")
@click.option("--dir", "data_dir", default=None, help="Directory to scan")
def data_stats_cmd(data_dir):
    """Show statistics for question data."""
    result = data_stats(data_dir=data_dir)
    _output(result)
    _session_record("data stats", result)


# ── Concept commands ────────────────────────────────────────────────────────

@cli.group()
def concept():
    """Manage concept graphs and seed concepts."""
    pass


@concept.command("extract")
@click.option("--input", "-i", "input_file", default=None, help="Input QA JSON file")
@click.option("--output", "-o", "output_dir", default=None, help="Output directory")
def concept_extract(input_file, output_dir):
    """Extract seed concepts from QA data."""
    result = extract_concepts(input_file=input_file, output_dir=output_dir)
    _output(result)
    _session_record("concept extract", result)


@concept.command("expand")
@click.option("--config", "-c", "config_file", default=None, help="Config file path")
@click.option("--stage", "stage", default="concept-expansion",
              type=click.Choice(["concept-analysis", "concept-extraction", "concept-expansion"]),
              help="Expansion stage to run")
def concept_expand(config_file, stage):
    """Run concept graph expansion via MemCube."""
    result = run_concept_expansion(config_file=config_file, stage=stage)
    _output(result)
    _session_record("concept expand", result)


@concept.command("info")
@click.option("--file", "-f", "graph_file", default=None, help="Concept graph JSON file")
def concept_info_cmd(graph_file):
    """Show concept graph information."""
    result = graph_info(graph_file=graph_file)
    _output(result)
    _session_record("concept info", result)


# ── QA commands ─────────────────────────────────────────────────────────────

@cli.group()
def qa():
    """Generate and manage QA pairs."""
    pass


@qa.command("generate")
@click.option("--graph", "-g", "graph_file", default=None, help="Concept graph file")
@click.option("--config", "-c", "config_file", default=None, help="Config file path")
def qa_generate_cmd(graph_file, config_file):
    """Generate QA pairs from concept graph."""
    result = generate_qa(graph_file=graph_file, config_file=config_file)
    _output(result)
    _session_record("qa generate", result)


@qa.command("pipeline")
@click.option("--config", "-c", "config_file", default=None, help="Config file path")
def qa_pipeline(config_file):
    """Run the full pipeline (concept expansion + QA generation)."""
    result = run_full_pipeline(config_file=config_file)
    _output(result)
    _session_record("qa pipeline", result)


@qa.command("stats")
@click.option("--file", "-f", "qa_file", default=None, help="QA JSON file")
def qa_stats_cmd(qa_file):
    """Show QA statistics."""
    result = qa_stats(qa_file=qa_file)
    _output(result)
    _session_record("qa stats", result)


# ── Config commands ─────────────────────────────────────────────────────────

@cli.group()
def config():
    """Manage configuration and environment."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    result = show_config()
    _output(result)


@config.command("check-env")
def config_check_env():
    """Check environment dependencies."""
    result = check_env()
    _output(result)


@config.command("test-api")
def config_test_api():
    """Test API configuration."""
    result = test_api()
    _output(result)


# ── Session commands ────────────────────────────────────────────────────────

@cli.group()
def session():
    """Manage persistent sessions."""
    pass


@session.command("list")
def session_list():
    """List all sessions."""
    sessions = Session.list_sessions()
    if not sessions:
        click.echo("No sessions found.")
    else:
        for s in sessions:
            click.echo(f"  {s['name']} (created: {s.get('created', 'N/A')}, commands: {s.get('commands', 0)})")


@session.command("info")
@click.option("--name", "-n", "session_name", default="default")
def session_info(session_name):
    """Show session info."""
    s = Session(name=session_name)
    _output({"status": "done", **s.info()})


@session.command("delete")
@click.option("--name", "-n", "session_name", required=True)
def session_delete(session_name):
    """Delete a session."""
    s = Session(name=session_name)
    s.delete()
    click.echo(f"Session '{session_name}' deleted.")


# ── REPL mode ───────────────────────────────────────────────────────────────

class CrawlerREPL(cmd.Cmd):
    """Interactive REPL for crawler commands."""

    intro = "CLI-Anything Crawler REPL. Type 'help' for commands, 'quit' to exit."
    prompt = "crawler> "

    def __init__(self, session_name=None):
        super().__init__()
        global _session
        if session_name:
            _session = Session(name=session_name)
        self._dispatch = {
            "crawl login": self._do_crawl_login,
            "crawl fetch": self._do_crawl_fetch,
            "crawl status": self._do_crawl_status,
            "data rewrite": self._do_data_rewrite,
            "data classify": self._do_data_classify,
            "data merge": self._do_data_merge,
            "data transform": self._do_data_transform,
            "data stats": self._do_data_stats,
            "concept extract": self._do_concept_extract,
            "concept expand": self._do_concept_expand,
            "concept info": self._do_concept_info,
            "qa generate": self._do_qa_generate,
            "qa pipeline": self._do_qa_pipeline,
            "qa stats": self._do_qa_stats,
            "config show": self._do_config_show,
            "config check-env": self._do_config_check_env,
            "config test-api": self._do_config_test_api,
            "session list": self._do_session_list,
        }

    def _parse_args(self, arg):
        """Parse arguments from REPL input."""
        parts = []
        current = ""
        in_quote = False
        quote_char = None
        for ch in arg:
            if ch in ('"', "'") and not in_quote:
                in_quote = True
                quote_char = ch
            elif ch == quote_char and in_quote:
                in_quote = False
                quote_char = None
            elif ch == " " and not in_quote:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += ch
        if current:
            parts.append(current)
        return parts

    def default(self, line):
        line = line.strip()
        if not line:
            return
        # Try to match a command
        for cmd_prefix, handler in self._dispatch.items():
            if line.startswith(cmd_prefix):
                args = line[len(cmd_prefix):].strip()
                handler(args)
                return
        click.echo(f"Unknown command: {line}. Type 'help' for available commands.")

    def _do_crawl_fetch(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p == "--subject" and i + 1 < len(parts):
                kwargs["subjects"] = (parts[i + 1],)
            elif p == "--all":
                kwargs["fetch_all_flag"] = True
            elif p == "--output" and i + 1 < len(parts):
                kwargs["output_dir"] = parts[i + 1]
            elif p == "--iterations" and i + 1 < len(parts):
                kwargs["iterations"] = int(parts[i + 1])
            elif p in ("--student-num", "-u") and i + 1 < len(parts):
                kwargs["student_num"] = parts[i + 1]
            elif p in ("--password", "-p") and i + 1 < len(parts):
                kwargs["password"] = parts[i + 1]
            elif p == "--student-id" and i + 1 < len(parts):
                kwargs["student_id"] = parts[i + 1]
        crawl_fetch.callback(**kwargs)

    def _do_crawl_login(self, args):
        parts = self._parse_args(args)
        student_num = password = None
        for i, p in enumerate(parts):
            if p in ("--student-num", "-u") and i + 1 < len(parts):
                student_num = parts[i + 1]
            elif p in ("--password", "-p") and i + 1 < len(parts):
                password = parts[i + 1]
        if not student_num:
            student_num = input("Student number: ")
        if not password:
            password = input("Password: ")
        crawl_login.callback(student_num=student_num, password=password)

    def _do_crawl_status(self, args):
        crawl_status_cmd.callback(data_dir=args or None)

    def _do_data_rewrite(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p in ("--input", "-i") and i + 1 < len(parts):
                kwargs["input_dir"] = parts[i + 1]
            elif p in ("--output", "-o") and i + 1 < len(parts):
                kwargs["output_dir"] = parts[i + 1]
        data_rewrite.callback(**kwargs)

    def _do_data_classify(self, args):
        data_classify.callback(input_dir=args or None)

    def _do_data_merge(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p == "--old" and i + 1 < len(parts):
                kwargs["old_dir"] = parts[i + 1]
            elif p == "--new" and i + 1 < len(parts):
                kwargs["new_dir"] = parts[i + 1]
            elif p in ("--output", "-o") and i + 1 < len(parts):
                kwargs["output_dir"] = parts[i + 1]
        data_merge.callback(**kwargs)

    def _do_data_transform(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p in ("--input", "-i") and i + 1 < len(parts):
                kwargs["input_dir"] = parts[i + 1]
            elif p in ("--output", "-o") and i + 1 < len(parts):
                kwargs["output_file"] = parts[i + 1]
        data_transform.callback(**kwargs)

    def _do_data_stats(self, args):
        data_stats_cmd.callback(data_dir=args or None)

    def _do_concept_extract(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p in ("--input", "-i") and i + 1 < len(parts):
                kwargs["input_file"] = parts[i + 1]
            elif p in ("--output", "-o") and i + 1 < len(parts):
                kwargs["output_dir"] = parts[i + 1]
        concept_extract.callback(**kwargs)

    def _do_concept_expand(self, args):
        parts = self._parse_args(args)
        kwargs = {"stage": "concept-expansion"}
        for i, p in enumerate(parts):
            if p in ("--config", "-c") and i + 1 < len(parts):
                kwargs["config_file"] = parts[i + 1]
            elif p == "--stage" and i + 1 < len(parts):
                kwargs["stage"] = parts[i + 1]
        concept_expand.callback(**kwargs)

    def _do_concept_info(self, args):
        concept_info_cmd.callback(graph_file=args or None)

    def _do_qa_generate(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p in ("--graph", "-g") and i + 1 < len(parts):
                kwargs["graph_file"] = parts[i + 1]
            elif p in ("--config", "-c") and i + 1 < len(parts):
                kwargs["config_file"] = parts[i + 1]
        qa_generate_cmd.callback(**kwargs)

    def _do_qa_pipeline(self, args):
        parts = self._parse_args(args)
        kwargs = {}
        for i, p in enumerate(parts):
            if p in ("--config", "-c") and i + 1 < len(parts):
                kwargs["config_file"] = parts[i + 1]
        qa_pipeline.callback(**kwargs)

    def _do_qa_stats(self, args):
        qa_stats_cmd.callback(qa_file=args or None)

    def _do_config_show(self, args):
        config_show.callback()

    def _do_config_check_env(self, args):
        config_check_env.callback()

    def _do_config_test_api(self, args):
        config_test_api.callback()

    def _do_session_list(self, args):
        session_list.callback()

    def do_quit(self, args):
        click.echo("Bye!")
        return True

    def do_exit(self, args):
        return self.do_quit(args)

    def do_json(self, args):
        """Toggle JSON output mode."""
        global _json_mode
        _json_mode = not _json_mode
        click.echo(f"JSON mode: {'ON' if _json_mode else 'OFF'}")

    def do_help(self, args):
        click.echo("\nAvailable commands:")
        click.echo("  crawl login --student-num NUM --password PASS")
        click.echo("  crawl fetch [--subject CODE | --all] [--output DIR] [--iterations N]")
        click.echo("  crawl status [--dir DIR]")
        click.echo("  data rewrite [--input DIR] [--output DIR]")
        click.echo("  data classify [--input DIR]")
        click.echo("  data merge [--old DIR] --new DIR [--output DIR]")
        click.echo("  data transform [--input DIR] [--output FILE]")
        click.echo("  data stats [--dir DIR]")
        click.echo("  concept extract [--input FILE] [--output DIR]")
        click.echo("  concept expand [--config FILE] [--stage STAGE]")
        click.echo("  concept info [--file FILE]")
        click.echo("  qa generate [--graph FILE] [--config FILE]")
        click.echo("  qa pipeline [--config FILE]")
        click.echo("  qa stats [--file FILE]")
        click.echo("  config show")
        click.echo("  config check-env")
        click.echo("  config test-api")
        click.echo("  session list")
        click.echo("  json                          Toggle JSON output mode")
        click.echo("  quit / exit")
        click.echo("")


@cli.command("repl")
@click.option("--session", "-s", "session_name", default=None, help="Session name")
def repl_cmd(session_name):
    """Start interactive REPL mode."""
    CrawlerREPL(session_name=session_name).cmdloop()


def main():
    cli()


if __name__ == "__main__":
    main()
