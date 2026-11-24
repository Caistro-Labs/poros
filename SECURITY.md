# Security

## Report a vulnerability

Please report vulnerabilities privately, never in a public issue: use
[Report a vulnerability](https://github.com/Caistro-Labs/poros/security/advisories/new) on the
Security tab, or email contact@caistrolabs.com. We reply within a week.

Include the Poros version or commit, how to reproduce it, what it lets someone do, and the model
or config files if they are safe to share. Say whether it exposes secrets, runs code, or loads a
model unsafely.

## Supported versions

Fixes go into the latest release and `main`.

## How releases are made

Releases are built and published by GitHub Actions from signed tags on `main`, with PyPI's
Trusted Publishing: no PyPI token exists, and PyPI shows each release's attestations. A tag
signed by a key outside `.github/allowed_signers` does not publish.

## Scope

Poros loads the Hugging Face models, tokenizers and datasets you choose, and model code when you
pass `trust_remote_code`. Treat those as untrusted unless you control them. Poros should never
log tokens, API keys, Hugging Face credentials or private dataset contents.
