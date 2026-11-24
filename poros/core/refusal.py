"""``Refusal``: a step that cannot go on, said once and shown by whoever asked.

A refusal is data: its contract code (``poros.core.errors.EXIT_CODES``), a short title, the
reason, the context lines, the command that fixes it and a note on that command. The preflight
(``poros.preflight``) and the CLI's verbs raise or return one; the CLI shows it as the ✕ screen
or the JSON error (``poros.cli.contract.refuse``, ``poros.cli.contract.refused``). Torch-free.
"""

from __future__ import annotations

from typing import Optional


class Refusal(Exception):
    """A resolve step that cannot go on: the ✕ screen, or the JSON error."""

    def __init__(
        self, code: str, title: str, reason: str, *, context=(), fix: str = "", note: str = "",
        docs: Optional[str] = None,
    ) -> None:
        super().__init__(reason)
        self.code, self.title, self.reason = code, title, reason
        self.context, self.fix, self.note, self.docs = list(context), fix, note, docs
