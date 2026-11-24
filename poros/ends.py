"""Where a chat's answers end, and whether a model ends them there.

A loss can hide a model that never stops: a base model trained on chats cut its held-out loss 30%
and never said its end-of-turn token. The chat template puts that token right after each answer,
and a model that has learned to stop gives it first there. A run's held-out pass and
``poros eval`` both read it here, from the logits a forward already has in hand.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

Read = List[Tuple[bool, float]]  # at each answer end of a row: the token is top, its probability


def turn_end(tok: Any) -> Optional[int]:
    """The id the chat template puts right after an assistant's words, if it is a special token
    of the tokenizer's own: what ends a turn. None without a template, or when that is text."""
    if not getattr(tok, "chat_template", None):
        return None
    from poros.core.template_guard import template_bound

    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    try:
        with template_bound():
            whole = tok.apply_chat_template(turns, tokenize=True, return_dict=False)
            asked = tok.apply_chat_template(turns[:1], tokenize=True, add_generation_prompt=True,
                                            return_dict=False)
        words = tok.encode("ok", add_special_tokens=False)
    except Exception:  # a template or tokenizer that cannot render this: no turn end to look for
        return None
    at = len(asked) + len(words)
    if whole[:len(asked)] != asked or whole[len(asked):at] != words or at >= len(whole):
        return None
    end = int(whole[at])
    return end if end in set(getattr(tok, "all_special_ids", []) or []) else None


class Ends:
    """The end-of-turn token, and per row of a Data where each answer ends: what a pass reads."""

    def __init__(self, token: int, at: List[List[int]]) -> None:
        self.token, self.at = token, at


def answer_ends(data: Any, tok: Any, end: int, *, train_on: str = "answers",
                source: Any = None, length: int = 2048) -> List[List[int]]:
    """Per row of ``data``, where each answer ends: in each run of answer positions, the last whose
    target is the end-of-turn token. A pair's are its chosen answer's. Chats trained on more than
    their answers find them in ``source``'s rows built for answers alone."""
    chosen = [item[0] if data.pairs else item for item in data.items]
    weights: List[Any] = [d.weights for d in chosen]
    if not data.pairs and train_on != "answers":
        want = {tuple(d.tokens) for d in chosen}
        alone = {tuple(d.tokens): d.weights for d in _answers_only(source, tok, length)
                 if tuple(d.tokens) in want}
        weights = [alone.get(tuple(d.tokens)) for d in chosen]
    return [_ends_in(d.target_tokens, w, end) if w else [] for d, w in zip(chosen, weights)]


def _answers_only(source: Any, tok: Any, length: int) -> list:
    from poros import data as pdata

    try:
        return pdata.chats(source, tok, train_on="answers", length=length).items
    except ValueError:  # every answer cut away: no end to look at
        return []


def _ends_in(targets: Sequence[int], weights: Sequence[float], end: int) -> List[int]:
    out, last, inside = [], None, False
    for i, (t, w) in enumerate(zip(targets, weights)):
        if w and t == end:
            last = i
        if inside and not w and last is not None:
            out.append(last)
            last = None
        inside = bool(w)
    return out + ([last] if last is not None else [])


def read(logits: Any, lp: Any, sub: Any, at: List[List[int]], end: int) -> List[Read]:
    """Per row of ``sub`` (a pair's chosen half), at each of its answer ends (``at``, by the row's
    id in its data): whether ``end`` is the model's top choice, and its probability."""
    out = []
    for j in range(len(sub) // 2 if sub.pairs else len(sub)):
        here = at[sub.ids[j]] if at else []
        top = logits[j, here].argmax(-1).tolist() if here else []
        out.append([(t == end, math.exp(float(lp[j, p]))) for t, p in zip(top, here)])
    return out


def ended(rows: Sequence[Read]) -> Tuple[int, int]:
    """(the rows whose every answer ends on the token, the rows with an answer end to read: an
    answer cut by the length has none)."""
    seen = [r for r in rows if r]
    return sum(all(top for top, _ in r) for r in seen), len(seen)
