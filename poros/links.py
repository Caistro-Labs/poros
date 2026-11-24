"""Where Poros's two documentation sets live: one place, read by the CLI and named by the docs.

- The documentation for people is on the Caistro Labs website, at ``DOCS_URL``. Its source is
  the repository's ``docs/`` folder, which ships with the package: ``poros docs`` serves it
  offline.
- The agent docs, for coding agents (exact facts, commands and JSON contracts), ship with the
  package too: ``AGENTS.md`` (in the repository, and installed as
  ``poros/agent_docs/AGENTS.md``) and ``poros agents`` (the guide, ``poros/agents.md``).
"""

DOCS_URL = "https://caistrolabs.com/docs/"  # the human docs, on the Caistro Labs website
ISSUES_URL = "https://github.com/Caistro-Labs/poros/issues"  # where a bug is reported


def doc_ref(doc: str) -> str:
    """A docs page an error names (``docs/limitations.md``, ``docs/moe.md#limitations``), as a
    reader of any install opens it: ``poros docs limitations``, ``poros docs moe limitations``
    (offline, from this install's copy of the docs for people). A web address, or anything else,
    is kept as it is."""
    import re

    m = re.fullmatch(r"(?:docs/)?([a-z0-9][a-z0-9-]{0,63})\.md(?:#([a-z0-9][a-z0-9-]*))?",
                     doc or "")
    if m is None:
        return doc
    page, anchor = m.groups()
    return " ".join(["poros docs", page, *(anchor.split("-") if anchor else [])])
