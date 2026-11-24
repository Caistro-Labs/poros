"""Structured errors for Poros.

Every error carries a type, message, remediation hint, and documentation URL
so that no failure is a dead end for the user.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# The CLI's stable error codes and the process exit code each maps to
# (docs/cli.md). One table, so a code can never exit two ways.
EXIT_CODES: dict[str, int] = {
    "E_INTERNAL": 1,
    "E_USAGE": 2,
    "E_CONFIG": 2,
    "E_CONFIRM_REQUIRED": 2,
    "E_NEEDS_INPUT": 2,
    "E_NOT_FOUND": 2,
    "E_OOM": 3,
    "E_OOM_PREDICTED": 3,
    "E_HOST_OOM_PREDICTED": 3,
    "E_ARCH_UNSUPPORTED": 4,
    "E_GUARANTEE_UNAVAILABLE": 4,
    "E_EXPORT_UNSUPPORTED": 4,
    "E_GPU_UNSUPPORTED": 4,
    "E_STEP_CHECK": 4,
    "E_NO_TORCH": 5,
    "E_NO_CUDA": 5,
    "E_DEPS": 5,
    "E_DISK": 5,
    "E_NETWORK": 5,
    "E_HUB_AUTH": 5,
    "E_STACK_UNSUPPORTED": 5,
    "E_MACHINE_NOT_READY": 5,
    "E_DATASET": 6,
    "E_DATASET_NOT_FOUND": 6,
    "E_NO_RESULTS": 6,
    "E_JOB_FAILED": 7,
    "E_JOB_CANCELLED": 7,
    "E_INTERRUPTED": 130,
}


# What a key looks like, wherever it turns up. A long run of word characters starts one match at
# most, so this stays linear on anyone's row. An sk- key is a long run of letters and digits, or
# names its issuer first (sk-proj-, sk-ant-): sk-learn-baseline-v2 is a folder's name.
KEY = re.compile(
    r"\bhf_[A-Za-z0-9]{16,}|\bsk-(?:(?:proj|ant|svcacct|admin|or|lf)-[\w-]{12,}|[A-Za-z0-9]{20,})"
    r"|\bAKIA[0-9A-Z]{12,}|\bAIza[\w-]{20,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{16,}|\bgithub_pat_\w{16,}|\bxox[abprs]-[\w-]{10,}"
    r"|\bnpm_[A-Za-z0-9]{16,}|\b[sr]k_(?:live|test)_\w{16,}|\bSG\.[\w-]{16,}\.[\w-]{16,}"
    r"|\bpypi-[\w-]{16,}|\bglpat-[\w-]{16,}|\bya29\.[\w-]{16,}|hooks\.slack\.com/services/\S{1,128}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----|(?<![\w-])eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{4,}"
    r"|\b[a-z][a-z0-9+.-]{0,16}://[^\s/:@]{1,256}:[^\s/@]{1,256}@")
NOT_CACHED = re.compile(r"couldn't connect|offline|outgoing traffic|cached (files|snapshot)|disk "
                        r"cache", re.I)  # transformers' and huggingface_hub's words for it
_ERROR_KEPT = 2000  # characters of an error written to a run's status


def no_keys(text: str) -> str:
    """``text`` with anything shaped like a key taken out."""
    return KEY.sub("[a key]", text)


def library_error(exc: BaseException) -> str:
    """All a library's exception says in a result, a job's error or a log ``runs`` reads: its type's
    name. Its text quotes whatever it was reading, and that was a token, once a folder's README said
    so. The whole of it goes to stderr, a job's log, for a person. What a person needs stays, in our
    words or the system's: that a file is not here, and which errno a disk gave."""
    if NOT_CACHED.search(str(exc)):
        return f"{type(exc).__name__}: not in the disk cache, and the Hub was not reached"
    if isinstance(exc, OSError) and isinstance(exc.errno, int):
        return f"{type(exc).__name__}: {os.strerror(exc.errno)}"
    return type(exc).__name__


def error_to_keep(error: object) -> object:
    """A run's or a job's error, as it is written to its status: where every later reader finds it,
    whichever process wrote it and whatever that process read."""
    if not isinstance(error, dict):
        return error
    return {k: no_keys(v)[:_ERROR_KEPT] if isinstance(v, str) else v for k, v in error.items()}


@dataclass
class PorosErrorInfo:
    """Structured payload attached to every Poros error."""

    error_type: str
    message: str
    remediation: str = ""
    doc_url: str = ""
    # Machine-applicable remediation: config field -> suggested value.
    fix: dict = field(default_factory=dict)


class PorosError(Exception):
    """Base exception for all Poros errors.

    ``code`` is the stable machine code (a key of ``EXIT_CODES``); subclasses
    override it. A bare PorosError is a refusal of how the API was used.
    """

    code = "E_USAGE"

    def __init__(
        self,
        message: str,
        *,
        remediation: str = "",
        doc_url: str = "",
        fix: dict | None = None,
    ) -> None:
        from poros.links import doc_ref

        self.info = PorosErrorInfo(
            error_type=type(self).__name__,
            message=message,
            remediation=remediation,
            doc_url=doc_ref(doc_url),  # "poros docs <page>": opens offline from any install
            fix=fix or {},
        )
        super().__init__(message)

    def __str__(self) -> str:
        # The remediation must be visible where the error actually surfaces.
        # Many Poros errors are raised on Python-API paths (PorosTrainer,
        # poros.Run) that never pass through the CLI's renderer -- a bare
        # message with the fix stored in an attribute nobody prints is a
        # refusal that does not teach. The CLI is unaffected: it renders
        # exc.info.message / .remediation itself.
        parts = [self.info.message]
        if self.info.remediation:
            parts.append(f"Remediation: {self.info.remediation}")
        if self.info.doc_url:
            parts.append(f"Docs: {self.info.doc_url}")
        return "\n\n".join(parts)


class PorosArchNotValidatedError(PorosError):
    """Raised when a model architecture has not been validated for blockwise parity."""

    code = "E_ARCH_UNSUPPORTED"

    def __init__(
        self,
        arch_type: str,
        supported: set[str] | None = None,
        remediation: str | None = None,
        *,
        unsupported: bool = False,
        moe: bool = False,
        selfcheck: str = "",
    ) -> None:
        # ``supported``: what this install trains (validated and user-asserted, and the MoE
        # families), not the validated set alone. ``unsupported``: a registry variant no loop
        # trains, which no setting lets through. ``moe``: a family validated on poros.moe, given
        # to the dense engine. ``selfcheck``: the startup parity self-check's figures, where it
        # failed.
        supported_str = ", ".join(sorted(supported)) if supported else "unknown"
        if unsupported:
            said = (f"Architecture '{arch_type}' is a variant neither the dense nor the MoE loop "
                    "trains.")
        elif moe:
            said = (f"Architecture '{arch_type}' trains on poros.moe, not on the dense blockwise "
                    "engine.")
        elif selfcheck:
            said = (f"The startup parity self-check failed for '{arch_type}' ({selfcheck}): the "
                    "streamed forward's loss is not the resident one's.")
        else:
            said = f"Architecture '{arch_type}' has not been validated for blockwise parity."
        super().__init__(
            message=f"{said} Architectures this install trains: {supported_str}.",
            remediation=remediation
            or (
                "Use a validated architecture, set blockwise_arch_check='off' to bypass "
                "(at your own risk), or open an architecture request issue to have it validated."
            ),
            doc_url="docs/models.md",
        )


class PorosOOMError(PorosError):
    """Raised when GPU memory is exhausted during blockwise training."""

    code = "E_OOM"

    def __init__(
        self,
        message: str = "GPU out of memory during blockwise training.",
        *,
        block_size: int | None = None,
        seq_len: int | None = None,
        remediation: str | None = None,
    ) -> None:
        self.block_size = block_size
        self.seq_len = seq_len
        parts = ["Try reducing block_size or seq_len."]
        fix: dict = {}
        if block_size is not None:
            fix["block_size"] = max(1, block_size // 2)
            parts.append(f"Current block_size={block_size}; try block_size={fix['block_size']}.")
        if seq_len is not None:
            fix["max_seq_length"] = max(128, seq_len // 2)
            parts.append(f"Current seq_len={seq_len}; try seq_len={fix['max_seq_length']}.")
        super().__init__(
            message=message,
            remediation=remediation or " ".join(parts),
            doc_url="docs/limitations.md",
            fix=fix,
        )


class PorosStepCheckError(PorosError):
    """Raised when a per-step check failed: the run's numbers cannot be trusted."""

    code = "E_STEP_CHECK"

    def __init__(self, message: str, *, witness: dict | None = None, **kwargs) -> None:
        self.witness = witness  # what the receipt records: mismatches, first layer, step
        super().__init__(message, **kwargs)


class PorosHostRamError(PorosError):
    """Raised before the download when a run needs more host RAM than is available.

    Its words come from one place, for every surface (the trainer from Python, poros train's
    screen and --json, the card's question, the agent verbs): ``head`` (what was refused),
    ``cause`` (why, in words; "" where only a planned figure says it) and ``advice`` (each step
    that can help this run, and only those). A surface adds only how to try anyway."""

    code = "E_HOST_OOM_PREDICTED"

    def __init__(self, model: str, need_gib: float, free_gib: float, basis: str, *,
                 at_least: bool = False, ssd_gib: float | None = None,
                 path: str | None = None, open_paths: tuple = ()) -> None:
        self.need_gib, self.free_gib = need_gib, free_gib
        self.model, self.path, self.at_least, self.basis = model, path, at_least, basis
        self.head = f"{model} needs more host RAM than the {free_gib} GiB available here"
        # ``at_least``: ``basis`` says, in words, what the run must hold at the least (the frozen
        # base in host RAM); any other basis carries the plan's figures and is not said
        self.cause = basis if at_least else ""
        advice = []
        if ssd_gib is not None:  # a MoE model on the RAM store whose SSD store the check passes
            advice.append("store: ssd keeps the experts in a .sss file on a local SSD instead, "
                          "which the check does not refuse for the host RAM here (poros train "
                          "--store ssd:/path/on/an/ssd, or store='ssd', store_path=...; the same "
                          "values, read from the disk)")
        advice.append("Run on a machine or container with more RAM free, or stop what is using "
                      "it")
        # ``open_paths``: the paths the check does not refuse for a run that set path: efficient
        # by hand (with the path left unset the check has tried them already; no path moves the
        # frozen base): named only where there is one
        if open_paths and not at_least:
            advice.append(" or ".join(f"path: {x}" for x in open_paths)
                          + " is not refused here: efficient holds its block inputs in host RAM")
        advice.append("poros recipes lists the runs certified on each machine")
        self.advice = advice
        super().__init__(
            message=(f"{self.head} (MemAvailable, capped by the container's memory limit)"
                     + (f", for {self.cause}" if self.cause else "")
                     + ". Refusing before anything is downloaded or loaded."),
            remediation=(_sentences(advice) + " To try anyway: poros train --force, or "
                         "PorosTrainer.from_pretrained(..., check_host_ram=False) "
                         "(poros.Run.create takes check_host_ram too)."),
            doc_url="docs/training.md#host-ram",
        )


def _sentences(parts: list) -> str:
    """Each piece of advice its own sentence, as it is written (a setting or a command may start
    one, in its own case)."""
    return " ".join(p if p.endswith(".") else p + "." for p in parts if p)


class PorosVramError(PorosError):
    """Raised before the download when the plan says the run cannot fit the GPU on any path it
    could take (a measured peak, or a run no larger's, past the free VRAM). Its message names no
    planned figure: ``basis`` keeps the plan's reason for code that reads it. ``head``, ``cause``
    and ``advice`` are its words for every surface, as PorosHostRamError's."""

    code = "E_OOM_PREDICTED"

    def __init__(self, model: str, gpu: str, basis: str, would: str = "",
                 any_path: bool = False, shortest: int | None = None) -> None:
        self.basis, self.would = basis, would
        self.model, self.gpu, self.any_path, self.shortest = model, gpu, any_path, shortest
        on = " on any path" if any_path else ""  # the path was the run's to choose: none fits
        self.head = f"{model} won't fit {gpu}{on}"
        # ``shortest``: refused at the shortest length the plan reads, so at any length
        self.cause = ("at these settings the run needs more memory than this GPU can give it"
                      if shortest is None else
                      f"even at max_seq_length {shortest}, the shortest the plan reads, so at "
                      "any length, the run needs more memory than this GPU can give it")
        # what the check found it does not refuse (fit.what_would_fit, mac.would_fit), else that
        # it found nothing: never a generic step that may not help this run
        self.advice = ([f"What to change: {would}"] if would else
                       ["The check found no setting here it does not refuse"])
        self.advice.append("poros recipes lists the runs certified on this card")
        super().__init__(
            message=f"{self.head}: {self.cause}. Refusing before anything is downloaded or loaded.",
            remediation=(_sentences(self.advice) + " To try anyway: poros train --force, or "
                         "PorosTrainer.from_pretrained(..., check_fit=False) (poros.Run.create "
                         "takes check_fit too)."),
            doc_url="docs/training.md#what-fits",
        )


class PorosDiskError(PorosError):
    """A disk problem Poros can name: no room, or a volume that is gone."""

    code = "E_DISK"


class PorosConfigError(PorosError):
    """Raised for invalid or inconsistent configuration."""

    code = "E_CONFIG"

    def __init__(
        self,
        message: str,
        *,
        remediation: str = "",
        doc_url: str = "",
    ) -> None:
        super().__init__(
            message=message,
            remediation=remediation
            or "Check the configuration documentation for valid options.",
            doc_url=doc_url or "docs/api.md",
        )


class PorosGpuUnsupportedError(PorosConfigError):
    """Raised before the download when this GPU is not one Poros supports for what the config
    asks (a card without native bf16, for bf16 compute). A PorosConfigError, with the code of the
    class it is: not supported (exit 4)."""

    code = "E_GPU_UNSUPPORTED"


class PorosTorchBuildError(PorosConfigError):
    """Raised before the download when the installed PyTorch build has no kernels for this GPU:
    an environment problem (exit 5), fixed by installing another build."""

    code = "E_STACK_UNSUPPORTED"


class PorosExportError(PorosError):
    """A merged export that cannot be written faithfully (``poros.train.export``)."""

    code = "E_EXPORT_UNSUPPORTED"


class PorosGuaranteeUnavailableError(PorosError):
    """Raised when the guarantee a run asks for cannot be given here: it refuses before any
    model loads rather than run at a weaker one."""

    code = "E_GUARANTEE_UNAVAILABLE"

    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(
            message=message,
            remediation=remediation or "Fix the listed condition(s), then run again.",
            doc_url="docs/limitations.md",
        )


class PorosStackUnsupportedError(PorosError):
    """Raised when the installed stack cannot run Poros safely."""

    code = "E_STACK_UNSUPPORTED"

    def __init__(self, message: str, *, remediation: str = "") -> None:
        super().__init__(
            message=message,
            remediation=remediation
            or "Run `poros machine` for a full environment report.",
            doc_url="docs/limitations.md",
        )


class PorosDataError(PorosConfigError):
    """Raised when the dataset is unreadable, malformed, or has no usable rows. Its remediation is
    one about the data, or none: never a config's."""

    code = "E_DATASET"

    def __init__(self, message: str, *, remediation: str = "", doc_url: str = "") -> None:
        PorosError.__init__(self, message, remediation=remediation,
                            doc_url=doc_url or "docs/training.md#data")


class PorosDatasetNotFoundError(PorosDataError):
    """Raised when the configured dataset does not exist."""

    code = "E_DATASET_NOT_FOUND"


class PorosCancelledError(PorosError):
    """Raised when ``train(should_stop=...)`` asks to stop: a checkpoint was
    written at the step boundary, so the run resumes bitwise from it."""

    code = "E_JOB_CANCELLED"

    def __init__(self, step: int, checkpoint: str) -> None:
        # ``step``: the last update's index, from 0, as the checkpoint's folder is named
        self.step, self.checkpoint = step, checkpoint
        done = step + 1
        super().__init__(
            f"cancelled after {done} step{'' if done == 1 else 's'}; {checkpoint} holds the run "
            "so far.",
            remediation="poros jobs resume <id> continues a job; poros train "
            "<config.yaml> --resume-from latest continues a foreground run.",
        )



# C0 and C1, what can steer a terminal, and category Cf, what a reader cannot see: tag characters,
# bidi overrides, zero-width marks. Display only: a row's own text trains as it is.
CONTROL = re.compile(
    "[\x00-\x1f\x7f-\x9f\u00ad\u0600-\u0605\u061c\u06dd\u070f\u0890\u0891\u08e2\u180e"
    "\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufff9-\ufffb\U000110bd\U000110cd"
    "\U00013430-\U0001343f\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0000-\U000e007f]")
