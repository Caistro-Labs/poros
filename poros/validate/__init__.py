"""The procedure a new model family goes through before Poros calls it validated, as code.

    load_gate      the streamed NF4 load against the standard one, a sha256 of every tensor
    training_gate  streamed against resident training, bit for bit, at every step
    tiny_route     the CPU rung of the CUDA route: a tiny random model of a config
    pack           what a pass leaves to submit: receipts, a README and the registry line

``scripts/streamed_load_gate.py`` and ``scripts/training_parity.py`` are thin command
lines over the first two. ``poros prove <model> --unvalidated`` runs both gates in one process.
Nothing here changes a registry status: a family becomes validated only by a committed artifact
(docs/models.md, "Submitting").
"""
