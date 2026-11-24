"""Poros report: HTML, Markdown, and leaderboard report generation.

Pure-python subpackage -- works WITHOUT torch installed.
"""

from poros.report.leaderboard import (
    build_leaderboard,
    leaderboard_to_json,
    leaderboard_to_markdown,
)
from poros.report.markdown import (
    generate_markdown_report,
    load_jsonl_records,
    render_markdown_string,
)

try:
    from poros.report.html import (
        generate_html_report,
        generate_report_from_jsonl,
    )
except ImportError:
    # jinja2 not installed -- HTML report generation unavailable
    pass

__all__ = [
    "generate_html_report",
    "generate_report_from_jsonl",
    "generate_markdown_report",
    "load_jsonl_records",
    "render_markdown_string",
    "build_leaderboard",
    "leaderboard_to_json",
    "leaderboard_to_markdown",
]
