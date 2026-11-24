"""Inkling 2 Mini (``thinkingmachines/Inkling-2-Mini``) for poros.moe: a looped MoE decoder.

- ``config``: the text configuration, read from the released config.json.
- ``modeling``: the decoder Poros carries (text only), from the official Apache-2.0
  implementation in transformers 5.24 (NOTICE).
- ``visits``: streaming its layer visits through the block engine.
- ``codec``: the released checkpoint's names and its NVFP4 / MXFP8 expert formats.
"""
