"""The plan: what a run needs, from measured runs on real cards. ``poros train`` and
``PorosTrainer`` decide from it before a download; no command prints its figures.

Every number comes from ``cells.json``: the figures of runs measured on real cards, one cell
per run. A figure measured for the same config on the same GPU is ``measured``; the same config
on another GPU gives ``estimated`` VRAM and host RAM, with the method stated; anything else is
``unknown``, never interpolated across models. Torch-free.

Each name lives in one module, and is imported from there (this package re-exports none):

============  ===================================================================================
``cells``     the receipts (``CELLS``, ``MAC_CELLS``) and the shapes that let one model's
              receipts plan another (``SHAPES``, ``shape_twin``); the one home of the receipt set
``configs``   a model's config.json (read, fetched under a cap) and the size of its download
``receipts``  which receipts a request reads (``Request``), and the bounds between them
``estimate``  what is estimated where no receipt measured
``make``      the plan itself (``make_plan``)
============  ===================================================================================

What a run decides from its plan (adapt, warn, refuse) is ``poros.preflight.fit``; the GPU is
``poros.core.card``; the plan's data models are ``poros.core.schemas``.
"""
