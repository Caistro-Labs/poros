"""A chat template is a program, and whoever published the model wrote it.

transformers runs it in jinja's sandbox, which keeps it from the machine and not from filling the
machine: {{ 'a' * 3000000000 }} and two nested ranges each went on until something killed them. So
wherever a template is applied, on every platform, what it renders has a length and a time, and a
power or a product is refused before it is computed. On Linux it also has a memory limit, on top of
what the process maps, for as long as it renders: RLIMIT_AS on a whole training process would fall
on CUDA's reservations of address space, and on a Mac the kernel does not enforce it.
"""

from __future__ import annotations

import contextlib
import signal
import sys
import threading
import time
from typing import Any, Iterable, Iterator

from poros.core.env import outside_words_out
from poros.core.errors import PorosDataError

RENDER_CAP = 256 << 20  # characters of one rendering; a row is at most 64 MiB of them
SECONDS = 60  # one template applied, tokenized too: a real one takes milliseconds
ROOM_BYTES = 3 << 30  # on top of what the process maps
ROOM_PER_CORE = 64 << 20  # a tokenizer's threads map stacks and arenas of their own
POWER_BITS = 1 << 24


class TemplateBound(PorosDataError):
    """A model's chat template asked for more than a rendering may take."""


def _past(what: str) -> TemplateBound:
    return TemplateBound(f"This model's chat template {what}: it is stopped, and nothing trains.",
                         remediation="Use a model whose template you trust.")


def _concat(chunks: Iterable[str]) -> str:
    """All of a rendering runs in this loop: jinja calls it to read the template's chunks."""
    parts, size, end = [], 0, time.monotonic() + SECONDS
    for chunk in chunks:
        size += len(chunk)
        if size > RENDER_CAP:
            raise _past(f"rendered past {RENDER_CAP >> 20} MiB")
        if time.monotonic() > end:
            raise _past(f"ran past {SECONDS} seconds")
        parts.append(chunk)
    return "".join(parts)


def _binop(self: Any, context: Any, operator: str, left: Any, right: Any) -> Any:
    if operator == "*":
        for sized, times in ((left, right), (right, left)):
            if isinstance(times, int) and hasattr(sized, "__len__"):
                if len(sized) * times > RENDER_CAP:
                    raise _past(f"asked for a value past {RENDER_CAP >> 20} MiB")
    elif isinstance(left, int) and isinstance(right, int) and right > 0:
        if left.bit_length() * right > POWER_BITS:
            raise _past("asked for a power past what a template may compute")
    return self.binop_table[operator](left, right)


def _arm() -> None:
    """Into jinja's sandbox itself, as transformers makes its own: a product and a power are then
    not folded when the template compiles, which is where they would have run."""
    try:
        from jinja2.sandbox import SandboxedEnvironment as Sandbox
    except ImportError:
        return
    Sandbox.concat = staticmethod(_concat)
    Sandbox.intercepted_binops = frozenset({"*", "**"})
    Sandbox.call_binop = _binop


@contextlib.contextmanager
def _deadline() -> Iterator[None]:
    """A template with no output is stopped here. Only the main thread gets a signal; elsewhere
    ``_concat`` holds the time, between what the template writes."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(*_: Any) -> None:
        raise _past(f"ran past {SECONDS} seconds")

    before = signal.signal(signal.SIGALRM, stop)
    left = signal.setitimer(signal.ITIMER_REAL, SECONDS)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *left)
        signal.signal(signal.SIGALRM, before)


@contextlib.contextmanager
def _memory_held() -> Iterator[None]:
    if sys.platform != "linux":
        yield
        return
    import os
    import resource

    with open("/proc/self/statm") as f:
        mapped = int(f.read().split()[0]) * resource.getpagesize()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    room = mapped + ROOM_BYTES + ROOM_PER_CORE * (os.cpu_count() or 1)
    resource.setrlimit(resource.RLIMIT_AS, (room if hard < 0 else min(room, hard), hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


@contextlib.contextmanager
def template_bound() -> Iterator[None]:
    """Around every ``apply_chat_template``. Under a tool or in a job, whose error a tool reads, a
    template's failure says its type: its text is whatever it said, or a row it repeats."""
    _arm()
    try:
        with _deadline(), _memory_held():
            yield
    except TemplateBound:
        raise
    except Exception as exc:
        if not outside_words_out():
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)  # a job's log, for a person
        raise ValueError(f"the model's chat template failed: {type(exc).__name__}") from None
