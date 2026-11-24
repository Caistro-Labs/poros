"""Poros core: blockwise residency engine, configuration, and determinism.

This subpackage contains the blockwise engine and its five parity invariants
(INV-1 through INV-5).

Modules:
    blockwise    -- JITFrozenBlockFunction, JITAdapterBlockFunction (INV-1)
    residency    -- BlockManager with record_stream safety (INV-2)
    determinism  -- enforce_determinism (INV-3)
    arch_registry-- the architecture registry + startup parity check (INV-4, INV-5)
    config       -- PorosConfig pydantic model
    errors       -- Structured error types
    events       -- JSONL event envelope
    env          -- Environment detection
"""
