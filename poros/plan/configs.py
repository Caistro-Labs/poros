"""A model's config.json, from a folder, a .json file or the Hub (that file only, under a cap),
and the size of its download: what the plan reads of a model before any weights. Torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from poros.core.env import under_agent
from poros.core.hub_files import SMALL_FILES
from poros.core.schemas import Figure

if TYPE_CHECKING:
    from poros.core.config import PorosConfig

CONFIG_CAP = 1 << 20  # the largest config.json read, from anywhere
UNSEEN_BYTES = 100 << 20  # what may come along, once the weights are here, with no one asked


class ConfigTooLarge(ValueError):
    """A Hub's config.json past the cap: it is never fetched."""


def download_size(cfg: "PorosConfig") -> Figure:
    """Bytes still to fetch: every file of the revision a load would fetch, not yet in the HF cache.
    With the weights here, what is left under UNSEEN_BYTES, a tokenizer's files, is not asked about:
    they come with any run. A repo's 5 GiB tokenizer.model is."""
    if Path(cfg.model_name_or_path).exists():
        return Figure(value=0.0, unit="GiB", source="local")
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache
        from huggingface_hub.utils import filter_repo_objects

        info = _token_only_if_needed(cfg.model_name_or_path, lambda token: HfApi().model_info(
            cfg.model_name_or_path, revision=cfg.model_revision, files_metadata=True, token=token))
    except Exception as exc:  # network, auth, offline: say so, don't guess
        return Figure(value=None, unit="GiB", source="unknown", why=_no_size(exc))
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")] or [
        s for s in info.siblings if s.rfilename.endswith(".bin")
    ]
    rest = [s for s in filter_repo_objects(info.siblings, allow_patterns=SMALL_FILES,
                                           key=lambda s: s.rfilename) if s not in weights]

    def missing(files: list) -> list:
        return [s for s in files if not isinstance(
            try_to_load_from_cache(cfg.model_name_or_path, s.rfilename, revision=info.sha), str)]

    # Nothing to fetch only when the weights the Hub lists are here. A repo that lists none, as one
    # whose index names a shard weights-00001-of-00001, or does not say a size, is not zero bytes.
    weights_gone, rest_gone = missing(weights), missing(rest)
    if not weights or any(not s.size for s in weights_gone) or any(
            s.size is None for s in rest_gone):
        return Figure(value=None, unit="GiB", source="unknown",
                      why="the Hub's list of files does not say what a load would fetch")
    fetched = sum(s.size for s in weights_gone + rest_gone)
    if not weights_gone and fetched < UNSEEN_BYTES:
        fetched = 0
    return Figure(value=max(round(fetched / 2**30, 2), 0.01) if fetched else 0.0, unit="GiB",
                  source="hub_metadata", method=f"revision {info.sha}")


def _no_size(exc: Exception) -> str:
    """Why the Hub did not say a download's size, in a person's words: never a class's name."""
    name = type(exc).__name__
    if name == "OfflineModeIsEnabled":
        return "the Hub was not asked: Hub access is off here (HF_HUB_OFFLINE is set)"
    if name == "GatedRepoError":
        return "its repository is gated: the Hub says its size once you have access and log in"
    if name == "RepositoryNotFoundError":
        return "the Hub has no repository of this name, or it is private"
    return "the Hub could not be reached"


def weights_on_hub(model: str, revision: Optional[str] = None) -> Optional[tuple[int, int]]:
    """(bytes of the weights the Hub lists for ``model`` at ``revision``, bytes of them not in
    the HF cache yet), from its metadata alone; None where it cannot say (a local folder, offline,
    no sizes listed). What a merged export reads (every shard) and would download first."""
    if Path(model).exists():
        return None
    try:
        from huggingface_hub import HfApi, try_to_load_from_cache

        info = _token_only_if_needed(model, lambda token: HfApi().model_info(
            model, revision=revision, files_metadata=True, token=token))
    except Exception:  # network, auth, offline: the caller says it could not tell
        return None
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    if not weights or any(not s.size for s in weights):
        return None
    gone = [s for s in weights if not isinstance(
        try_to_load_from_cache(model, s.rfilename, revision=info.sha), str)]
    return sum(s.size for s in weights), sum(s.size for s in gone)


def _token_only_if_needed(model: str, call: Any) -> Any:
    """``call(token)``, first with none: a public repo needs none, and its owner learns nothing of
    whoever asked. The token only for a repo the Hub says needs it, and one a person named: an
    agent's call may be anyone's words, and it gets the token for a catalog model alone."""
    from huggingface_hub.utils import RepositoryNotFoundError  # gated is one of these

    from poros.core.env import quiet_hub_notice

    quiet_hub_notice()  # asked with no token on purpose: the Hub's notice of it is not news
    try:
        return call(False)
    except RepositoryNotFoundError:
        from poros.catalog import names

        if under_agent() and not names(model):
            raise
        return call(None)


def fetch_config(model: str, revision: Optional[str] = None) -> Path:
    """A Hub id's config.json, that file only, with no one asked: from the cache if it holds it,
    else fetched once its size says it can be read. ConfigTooLarge past the cap; what the Hub raises
    otherwise, a gated repo with no token here for one."""
    from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
    from huggingface_hub.utils import RepositoryNotFoundError

    cached = try_to_load_from_cache(model, "config.json", revision=revision)
    if isinstance(cached, str):
        return Path(cached)

    def fetched(token: Any) -> Path:
        try:
            info = HfApi().model_info(model, revision=revision, files_metadata=True, token=token)
        except RepositoryNotFoundError:
            raise
        except Exception:  # offline or the network: the download below says which
            info = None
        size = next((s.size for s in (info.siblings or []) if s.rfilename == "config.json"), None
                    ) if info else None
        if (size or 0) > CONFIG_CAP:
            raise ConfigTooLarge(f"its config.json is over {CONFIG_CAP:,} bytes")
        return Path(hf_hub_download(repo_id=model, filename="config.json", revision=revision,
                                    token=token))

    return _token_only_if_needed(model, fetched)


def read_config(path: Path) -> Optional[dict]:
    """A regular file's JSON table under the cap, or None: a config may be anyone's."""
    try:
        if not path.is_file() or path.stat().st_size > CONFIG_CAP:
            return None
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None
    return raw if isinstance(raw, dict) else None


def read_model_config(model: str, revision: Optional[str] = None) -> Optional[dict]:
    """config.json of a model directory, a .json path or a Hub id (that file only)."""
    try:
        path = Path(model)
        if path.is_dir():
            path = path / "config.json"
        elif not (path.is_file() and path.suffix == ".json"):
            path = fetch_config(model, revision)
        return read_config(path)
    except Exception:  # offline, gated, missing: no figure, never a refusal
        return None


def cached_model_config(model: str) -> Optional[dict]:
    """config.json of a model directory, or of a Hub id from the Hugging Face cache alone: no
    network, so a list of every catalog model costs nothing for the ones not downloaded. A Hub id
    the cache has no config.json of: the facts Poros ships for it (``shipped_config``), if any."""
    try:
        path = Path(model)
        if path.is_dir():
            return read_config(path / "config.json")
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(model, "config.json")
        return read_config(Path(hit)) if isinstance(hit, str) else shipped_config(model)
    except Exception:
        return None


# The config.json fields the plan reads (scripts/build_plan_configs.py writes config_facts.json
# with these, for the models the built-in recipes name): sizes, the MoE layout, the family, and
# the model's own random draws (any field ending in dropout, pdrop or jitter_noise).
PLAN_KEYS = ("model_type", "text_config", "num_hidden_layers", "hidden_size", "intermediate_size",
             "vocab_size", "num_attention_heads", "num_key_value_heads", "head_dim",
             "tie_word_embeddings", "layer_types", "quantization_config", "hidden_act",
             "hidden_activation", "qk_nope_head_dim", "qk_rope_head_dim", "num_experts",
             "num_local_experts", "n_routed_experts", "num_experts_per_tok", "top_k_experts",
             "moe_intermediate_size", "shared_expert_intermediate_size", "n_shared_experts",
             "decoder_sparse_step", "mlp_only_layers", "first_k_dense_replace",
             "mlp_layer_types", "enable_moe_block", "output_router_logits",
             "router_aux_loss_coef")


def shipped_config(model: str) -> Optional[dict]:
    """The config.json facts the plan reads (PLAN_KEYS), as Poros ships them for the models its
    built-in recipes name: what a fresh install plans from before any download. None for any
    other model. Never what a run, a fit or a gate reads: those read the model's own file."""
    from importlib import resources

    raw = json.loads(resources.files("poros.plan").joinpath("config_facts.json").read_text())
    hit = raw["models"].get(model)
    return dict(hit["fields"]) if hit else None
