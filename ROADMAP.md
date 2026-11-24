# Roadmap

What comes next for Poros. No dates; each item ships when streamed training still matches fully
resident training.

- **More model families**, dense and MoE, each checked streamed against fully resident before it
  is supported.
- **Multi-GPU training**, data-parallel: each GPU streams its own copy of the base and the
  adapter gradients are synced.
- **Faster streaming**: fused kernels for more layers and families, and deeper prefetch.
- **A wider GPU matrix**: the supported families checked across consumer and datacenter cards.
- **Published benchmarks** for more cards and model sizes, each from a measured run.
- **Harder testing**: fixture-style architecture tests for masks, rotary embeddings and layer
  calls; soak tests of long runs, resume, export and adapter loading; clean-room install checks
  for common CUDA and bitsandbytes setups.
