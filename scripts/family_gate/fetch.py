#!/usr/bin/env python3
"""fetch.py OUT_DIR REPO REVISION MODE [LAYERS] -- one family gate model, public Hub files only, no
token.

MODE:
  snapshot   the whole repo at REVISION (huggingface_hub.snapshot_download into HF_HOME); OUT_DIR
             becomes a symlink to the snapshot.
  trunc      the first LAYERS decoder layers of REPO with their real weights, in the repo's own
             format (a multimodal wrapper keeps its towers and key names), fetched by HTTP range
             requests so only the tensors kept cross the network: every tensor that is not a decoder
             layer's (embedding, final norm, head, vision/audio towers), and decoder layers
             0..LAYERS-1. The config's num_hidden_layers and every per-layer list are cut to LAYERS.
             Multi-token-prediction heads (``mtp.*``) are dropped: transformers never loads them.
  text       as trunc, then the text backbone alone as its CausalLM: ``model.language_model.*`` ->
             ``model.*``, towers dropped, config = the text config (its own model_type, *_text).

Writes OUT_DIR/model.safetensors, config.json, the tokenizer files, and OUT_DIR/fetch.json: the
repo, the revision, the mode, and for trunc/text every kept tensor's shard and byte range, and the
sha256 of model.safetensors. Idempotent: an OUT_DIR whose fetch.json matches the request is left as
it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request

out, repo, rev, mode = sys.argv[1:5]
layers = int(sys.argv[5]) if len(sys.argv) > 5 else 0
want = {"repo": repo, "revision": rev, "mode": mode, "layers": layers}
UA = {"User-Agent": "poros-family-gate"}


def done() -> bool:
    f = out.rstrip("/") + ".fetch.json" if mode == "snapshot" else os.path.join(out, "fetch.json")
    try:
        rec = json.load(open(f))
    except (OSError, ValueError):
        return False
    return all(rec.get(k) == v for k, v in want.items()) and rec.get("complete") is True


if done():
    print("FETCH cached", out)
    sys.exit(0)

if mode == "snapshot":
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo,
        revision=rev,
        token=False,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.txt",
            "*.jinja",
            "*.tiktoken",
            "tokenizer*",
        ],
    )
    if os.path.islink(out):
        os.unlink(out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.symlink(path, out)
    # fetch.json beside the link (the snapshot itself is HF's cache, never written to)
    json.dump({**want, "path": path, "complete": True}, open(out.rstrip("/") + ".fetch.json", "w"))
    print("FETCH snapshot", out, "->", path)
    sys.exit(0)

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

URL = f"https://huggingface.co/{repo}/resolve/{rev}/{{}}"
DT = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F64": torch.float64,
}


def get(name: str, start: int | None = None, end: int | None = None) -> bytes:
    for attempt in range(6):
        req = urllib.request.Request(URL.format(name), headers=dict(UA))
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end - 1}")
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
            if start is not None and len(data) != end - start:
                raise IOError(f"short read {len(data)} of {end - start}")
            return data
        except Exception as exc:  # network: retry with backoff
            print(f"retry {attempt} {name} {start}-{end}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise SystemExit(f"could not fetch {name}")


def files() -> list[str]:
    req = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo}/revision/{rev}", headers=UA
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return [s["rfilename"] for s in json.load(r).get("siblings", [])]


DECODER = re.compile(r"^(model\.language_model\.|model\.|language_model\.model\.)layers\.(\d+)\.")
TOWERS = ("vision", "audio", "visual", "multi_modal_projector", "embed_vision", "embed_audio")

LOCAL = os.environ.get("FG_TEXT_FROM") if mode == "text" else None
if LOCAL:  # text from a complete trunc dir of the same repo, revision and layers: no network
    lrec = json.load(open(os.path.join(LOCAL, "fetch.json")))
    if not (
        lrec.get("complete")
        and lrec.get("mode") == "trunc"
        and all(lrec.get(k) == want[k] for k in ("repo", "revision", "layers"))
    ):
        raise SystemExit(f"FG_TEXT_FROM={LOCAL} is not a complete trunc of {want}")
    import shutil

    from safetensors.torch import load_file

    need = os.path.getsize(os.path.join(LOCAL, "model.safetensors"))
    if (
        shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free - need
        < float(os.environ.get("FG_MIN_FREE_GB", "15")) * 1e9
    ):
        raise SystemExit(
            f"FETCH refused: {need / 1e9:.1f} GB to write would leave under FG_MIN_FREE_GB"
        )
    tensors = load_file(os.path.join(LOCAL, "model.safetensors"))
    cfg = json.load(open(os.path.join(LOCAL, "config.json")))  # already cut
    sources = {"from_trunc": LOCAL, "trunc_sha256": lrec.get("sha256")}
    listing = os.listdir(LOCAL)
if not LOCAL:
    cfg = json.loads(get("config.json"))
    listing = files()
    if "model.safetensors.index.json" in listing:
        index = json.loads(get("model.safetensors.index.json"))["weight_map"]
    else:
        index = None
    shards = sorted(set(index.values())) if index else ["model.safetensors"]
    headers = {}
    for shard in shards:
        n = struct.unpack("<Q", get(shard, 0, 8))[0]
        headers[shard] = (8 + n, json.loads(get(shard, 8, 8 + n)))
    if index is None:
        index = {
            k: "model.safetensors" for k in headers["model.safetensors"][1] if k != "__metadata__"
        }

    def kept(k: str) -> bool:
        if k.startswith("mtp.") or ".mtp." in k:
            return False
        m = DECODER.match(k)
        if m:
            return int(m.group(2)) < layers
        return True

    keep = {k: v for k, v in index.items() if kept(k)}
    need = sum(
        headers[v][1][k]["data_offsets"][1] - headers[v][1][k]["data_offsets"][0]
        for k, v in keep.items()
    )
    import shutil

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
    floor = float(os.environ.get("FG_MIN_FREE_GB", "15")) * 1e9
    if free - need < floor:
        raise SystemExit(
            f"FETCH refused: {need / 1e9:.1f} GB to write leaves {(free - need) / 1e9:.1f} GB "
            f"free, under FG_MIN_FREE_GB={floor / 1e9:.0f}"
        )
    # one range request per run of adjacent kept tensors in a shard (gaps under 4 MiB are read
    # through)
    tensors, sources = {}, {}
    for shard in shards:
        base, h = headers[shard]
        names = sorted(
            (k for k, s in keep.items() if s == shard), key=lambda k: h[k]["data_offsets"][0]
        )
        runs: list[list[str]] = []
        for k in names:
            s, e = h[k]["data_offsets"]
            if (
                runs
                and s - h[runs[-1][-1]]["data_offsets"][1] < (4 << 20)
                and e - h[runs[-1][0]]["data_offsets"][0] < (1 << 30)
            ):
                runs[-1].append(k)
            else:
                runs.append([k])
        for run in runs:
            r0 = h[run[0]]["data_offsets"][0]
            r1 = h[run[-1]]["data_offsets"][1]
            raw = get(shard, base + r0, base + r1)
            for k in run:
                meta = h[k]
                s, e = meta["data_offsets"]
                buf = bytearray(raw[s - r0 : e - r0])
                t = (
                    torch.frombuffer(buf, dtype=DT[meta["dtype"]]).reshape(meta["shape"]).clone()
                    if e > s
                    else torch.empty(meta["shape"], dtype=DT[meta["dtype"]])
                )
                tensors[k] = t
                sources[k] = {
                    "shard": shard,
                    "bytes": [base + s, base + e],
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                }
            print(f"{shard}: {len(run)} tensors, {(r1 - r0) / 2**20:.1f} MiB", flush=True)

    def cut(c: dict, n_old: int) -> dict:
        c = dict(c)
        for key, val in list(c.items()):
            if isinstance(val, list) and len(val) == n_old and key != "architectures":
                c[key] = val[:layers]
        c["num_hidden_layers"] = layers
        if isinstance(c.get("mlp_only_layers"), list):
            c["mlp_only_layers"] = [i for i in c["mlp_only_layers"] if i < layers]
        return c

    if "text_config" in cfg:
        n_old = cfg["text_config"]["num_hidden_layers"]
        cfg["text_config"] = cut(cfg["text_config"], n_old)
    else:
        cfg = cut(cfg, cfg["num_hidden_layers"])

if mode == "text":
    if "text_config" not in cfg:
        raise SystemExit(f"{repo} has no text_config: use mode trunc")
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    text = dict(cfg["text_config"])
    arch = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[text["model_type"]]
    text["architectures"] = [arch]
    for k in ("tie_word_embeddings", "torch_dtype", "dtype"):
        if k in cfg and k not in text:
            text[k] = cfg[k]
    renamed = {}
    for k, t in tensors.items():
        if k.startswith("model.language_model."):
            renamed["model." + k[len("model.language_model.") :]] = t
        elif k.startswith("lm_head."):
            renamed[k] = t
        elif any(x in k for x in TOWERS):
            continue
        else:
            raise SystemExit(f"text mode: no place for {k}")
    tensors, cfg = renamed, text

os.makedirs(out, exist_ok=True)
fp = os.path.join(out, "model.safetensors")
save_file(tensors, fp, metadata={"format": "pt"})
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
TOK = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "generation_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
)
for f in TOK:
    if f in listing:
        data = open(os.path.join(LOCAL, f), "rb").read() if LOCAL else get(f)
        open(os.path.join(out, f), "wb").write(data)
h = hashlib.sha256()
with open(fp, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
rec = {
    **want,
    "model_type": cfg.get("model_type"),
    "tensors": len(tensors),
    "params": sum(t.numel() for t in tensors.values()),
    "sha256": h.hexdigest(),
    "sources": sources,
    "complete": True,
}
json.dump(rec, open(os.path.join(out, "fetch.json"), "w"), indent=1)
print("FETCH wrote", out, cfg.get("model_type"), rec["params"], "params sha256", rec["sha256"][:16])
