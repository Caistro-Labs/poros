"""Test harness only: emulate another card class's choices on the card at hand, for the one check
of scripts/xcard_check.sh that needs it (refuse-bf16-emu75). Imported explicitly by that check's
Python before Poros; never put on PYTHONPATH as a sitecustomize, so a system's or a venv's own
sitecustomize still runs everywhere else.

POROS_XCARD_CAP="7.5": torch.cuda.get_device_capability answers (7, 5) to every caller in that
process, so Poros's choosers (poros.core.card) decide as on a Turing card.
"""
import os

_cap = os.environ.get("POROS_XCARD_CAP")
if _cap:
    import torch

    _fake = tuple(int(x) for x in _cap.split("."))
    torch.cuda.get_device_capability = lambda device=None: _fake
