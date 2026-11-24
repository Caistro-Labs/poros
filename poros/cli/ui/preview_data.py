"""The data a preview trains on (POROS_DEV_SIMULATE): made-up chats for SFT and made-up preference
pairs for DPO (and IPO, KTO, ORPO, SimPO), each big enough that rows are held out, so a simulated
run scores every checkpoint as a real one does. Written once to the cache, offered only while
previewing, and plain .jsonl you can open and read. A user's own data is never touched."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

VERSION = "2"  # bump when the rows change: an older cached file is written again
CHATS, PAIRS = "preview-chats.jsonl", "preview-pairs.jsonl"
CHAT_ROWS, PAIR_ROWS = 600, 400  # 5% held out, at most the cap: every checkpoint is scored

_SYSTEM = ("You are the support assistant for Lumen, a note-taking app. Answer in two or three "
           "plain sentences, and say exactly where in the app a setting lives.")

# (topic, question, a good answer, a worse one: vague, wrong or off-voice)
_QA: List[Tuple[str, str, str, str]] = [
    ("sync", "My notes aren't syncing between my laptop and phone.",
     "Open Settings → Account on both devices and check they show the same email. If they do, "
     "pull down on the note list on your phone to force a sync.",
     "Syncing can fail for many reasons. Try again later."),
    ("export", "How do I export a notebook as PDF?",
     "Open the notebook, then choose File → Export → PDF. Each note becomes a page, in the "
     "order the sidebar shows.",
     "You can't export notebooks, only single notes, and only as plain text."),
    ("search", "Search doesn't find words inside my scanned PDFs.",
     "Scanned pages need text recognition first: open the PDF, choose ⋯ → Recognize text, and "
     "search finds its words once the progress bar finishes.",
     "Search only works on typed text. That's just how it is."),
    ("shortcut", "Is there a shortcut for a new note?",
     "Yes: ⌘N on a Mac and Ctrl+N on Windows. You can change it under Settings → Shortcuts.",
     "There are lots of shortcuts! Check the menus, they're all over the place."),
    ("trash", "I deleted a note by accident. Can I get it back?",
     "Deleted notes stay in Trash for 30 days. Open Trash at the bottom of the sidebar, select "
     "the note, and choose Restore.",
     "Deleted means deleted, sorry. You should be more careful next time."),
    ("share", "How do I share a note with someone who doesn't use Lumen?",
     "Open the note and choose Share → Copy link. Anyone with the link can read it in a "
     "browser; turn on Allow editing only for people who have accounts.",
     "Email them a screenshot of it."),
    ("offline", "Can I use Lumen without internet?",
     "Yes. Everything you have opened once stays on the device; changes made offline sync the "
     "next time you connect.",
     "No, Lumen needs a connection at all times."),
    ("tags", "What's the difference between tags and notebooks?",
     "A note lives in one notebook but can carry many tags. Use notebooks for where things "
     "belong and tags for what they are about.",
     "They're basically the same thing, use whichever you like."),
    ("price", "Will I lose my notes if I cancel Pro?",
     "No. Your notes stay and remain editable; only Pro features such as version history and "
     "large attachments stop working.",
     "Yes, cancelling deletes your account and all its notes."),
    ("dark", "How do I turn on dark mode?",
     "Go to Settings → Appearance and pick Dark, or System to follow your computer.",
     "Dark mode is coming soon, maybe."),
    ("history", "Can I see an older version of a note?",
     "With Pro, open the note and choose ⋯ → Version history. Pick a version to preview it, "
     "then Restore to bring it back.",
     "Lumen doesn't keep old versions of anything."),
    ("import", "How do I bring my notes over from another app?",
     "Choose File → Import and pick the other app's export: Markdown folders, Evernote .enex "
     "and plain text all keep their titles and dates.",
     "Copy and paste each note one at a time."),
]

_FOLLOW = [
    ("Thanks, that worked. Can I make it do that automatically?",
     "Yes: turn on Settings → Account → Sync in background, and it syncs every few minutes."),
    ("Where do I find that on my phone?",
     "On the phone it is the same place: tap your avatar, then Settings."),
    ("Does this cost extra?", "No, it is included on every plan, free ones too."),
]

_OPENERS = ["", "Hi! ", "Quick question: ", "Hey, ", "Hello, I need help. ", "Sorry if this is "
            "obvious, but "]


def _ask(rng: random.Random, q: str) -> str:
    opener = rng.choice(_OPENERS)
    return opener + (q[0].lower() + q[1:] if opener.endswith(("but ", ": ")) else q)


def _chats(rng: random.Random) -> List[Dict]:
    rows = []
    for i in range(CHAT_ROWS):
        _, q, good, _ = _QA[i % len(_QA)]
        turns = [{"role": "system", "content": _SYSTEM}] if rng.random() < 0.5 else []
        turns += [{"role": "user", "content": _ask(rng, q)},
                  {"role": "assistant", "content": good}]
        if rng.random() < 0.3:  # some go on a turn
            fq, fa = rng.choice(_FOLLOW)
            turns += [{"role": "user", "content": fq}, {"role": "assistant", "content": fa}]
        rows.append({"messages": turns})
    return rows


def _pairs(rng: random.Random) -> List[Dict]:
    rows = []
    for i in range(PAIR_ROWS):
        _, q, good, worse = _QA[i % len(_QA)]
        rows.append({"prompt": _ask(rng, q), "chosen": good, "rejected": worse})
    return rows


def _root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "poros" / "preview"


def _write(name: str, rows: List[Dict]) -> Path:
    out = _root() / name
    stamp = out.with_name(out.name + ".version")
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == VERSION:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(out)
    stamp.write_text(VERSION + "\n")
    return out


def _both() -> Tuple[Path, Path]:
    """Both files, written together: a path typed from one finds the other there too."""
    return _write(CHATS, _chats(random.Random(11))), _write(PAIRS, _pairs(random.Random(13)))


def chats() -> Path:
    """Support chats, some with a system prompt, some two turns long: SFT."""
    return _both()[0]


def pairs() -> Path:
    """The same questions, each with a better and a worse answer: DPO and the other pair loops."""
    return _both()[1]
