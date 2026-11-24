"""`poros prove` on a fixture Mac: real child processes through the Mac path on the CPU, the
comparison in ``poros.metal.verify``, and a screen with none of the machinery in it."""

import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")  # the fixture Mac trains (tests.metal.nf4_tiny): the training stack

from poros import runs  # noqa: E402
from poros.cli import app, contract  # noqa: E402
from poros.cli.context import Context  # noqa: E402
from poros.cli.ui.text import plain  # noqa: E402
from poros.cli.verbs.prove import MacProveOutput, ProveAction  # noqa: E402
from poros.metal import prove as mac_prove  # noqa: E402
from poros.metal import table  # noqa: E402
from tests.conftest import needs_metal_on_a_mac  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

M4 = table.plan_gpu("apple-m4-10gpu-16gb")
BANNED = re.compile(
    r"resident|fit line|window|witness|validated|covered|unproven|cuda|vram|bitsandbytes|host ram",
    re.IGNORECASE,
)
ROOT = Path(__file__).resolve().parents[2]
DOUBLE = """
import os, sys
sys.path.insert(0, {scripts!r})
import arms, torch
import poros.train.trainer as trainer
from poros.metal import prove
from poros.metal.capture import TranscriptWriter
trainer.PorosTrainer = arms.CpuDouble
torch.set_num_threads(1)
if os.environ.get("BREAK_ARM") in sys.argv:
    final = TranscriptWriter.final
    def wrong(self, microsteps, adapter, adam, scheduler):
        final(self, microsteps, adapter={{**adapter, "x": torch.ones(1)}}, adam=adam,
              scheduler=scheduler)
    TranscriptWriter.final = wrong
sys.exit(prove.main())
""".format(scripts=str(ROOT / "scripts" / "metal"))


@pytest.fixture
def mac(tmp_path, monkeypatch):
    base = tmp_path / "tiny-qwen2"
    tiny.write_checkpoint(base, tiny.tiny_config())
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setattr(mac_prove, "CHILD", [sys.executable, "-c", DOUBLE])
    monkeypatch.setattr(mac_prove, "store_for", lambda model: str(tmp_path / "store"))
    monkeypatch.setattr("poros.cli.sheets.listings.largest_certified", lambda ctx: None)
    row = type("Row", (), {"tier": "ram", "method": ""})
    monkeypatch.setattr(table, "pick", lambda model, gpu, **kw: row)
    return ProveAction(model=str(base), steps=3, seq_len=16)


CTX = Context(fixed_gpu=M4, fixed_cached=frozenset(), fixed_interactive=False)


def _screen(out) -> str:
    return "\n".join(plain(line) for line in mac_prove.view(out, 100))


@needs_metal_on_a_mac
def test_ram_against_ssd_in_separate_processes_is_identical_with_a_receipt(mac):
    out = mac_prove.run(mac, CTX)
    assert out.differs == [] and out.identical and out.exit_code == 0
    assert out.method == "ram-vs-ssd" and out.resident.block_size == 1
    assert out.witness == {"checked": 24, "expected": 24, "mismatches": 0}
    stored = json.loads(Path(out.receipt).read_text())
    assert stored["schema"] == "poros.prove.v1" and stored["digest"] and stored["method"]
    assert {"chip", "gpu_cores", "macos_build", "shader_sha"} <= stored["env"].keys()
    gate = json.loads((runs.home() / "prove" / Path(out.receipt).stem[6:] / "receipt.json")
                      .read_text())
    assert gate["pass"] is True and gate["gate"] == "prove" and gate["probe"] is None
    assert [a["name"] for a in gate["arms"]] == ["ram-loop", "ssd"]
    assert gate["arms"][0]["digests"] == gate["arms"][1]["digests"]

    screen = _screen(out)
    assert "identical" in screen and "memory, RAM" in screen and "memory, SSD" in screen
    assert not BANNED.search(screen)


@needs_metal_on_a_mac
def test_a_rerun_of_a_receipt_reproduces_it_and_names_a_changed_pin(mac):
    first = mac_prove.run(mac, CTX)
    stored = json.loads(Path(first.receipt).read_text())
    stored["env"]["macos_build"] = "OLD"
    Path(first.receipt).write_text(json.dumps(stored))
    again = mac_prove.run(mac.model_copy(update={"receipt": first.receipt}), CTX)
    assert again.reproduced is True and list(again.pinned_differs) == ["macos_build"]
    assert "OLD" in _screen(again) and not BANNED.search(_screen(again))


@needs_metal_on_a_mac
def test_a_difference_reads_as_a_failure_in_plain_words(mac, monkeypatch):
    monkeypatch.setenv("BREAK_ARM", "ssd")
    out = mac_prove.run(mac, CTX)
    assert not out.identical and out.exit_code == 1
    assert any("differ on adapter" in f for f in out.differs)
    screen = _screen(out)
    assert "Not identical" in screen and not BANNED.search(screen)


@needs_metal_on_a_mac
def test_too_large_for_ram_runs_from_ssd_twice(mac, monkeypatch):
    row = type("Row", (), {"tier": "ssd", "method": ""})
    monkeypatch.setattr(table, "pick", lambda model, gpu, **kw: row)
    out = mac_prove.run(mac, CTX)
    assert out.identical and out.method == "ssd-twice", out.differs
    assert "SSD" in _screen(out) and not BANNED.search(_screen(out))


def test_a_refusal_in_an_arm_is_a_refusal_with_its_fix(mac, monkeypatch):
    from poros.core.refusal import Refusal

    monkeypatch.setattr(mac_prove, "CHILD", [sys.executable, "-c", """
import poros.core.errors as e
import sys
from poros.metal import prove
def refuse(a):
    raise e.PorosOOMError("out of memory", remediation="close something")
prove._run_arm = refuse
sys.exit(prove.main())
"""])
    with pytest.raises(Refusal) as hit:
        mac_prove.run(mac, CTX)
    assert hit.value.code == "E_OOM" and "out of memory" in hit.value.reason


@pytest.fixture
def words(tmp_path, monkeypatch):
    import poros.plan.configs
    from poros.cli.schemas import Figure

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("POROS_OUTPUT", "json")
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    monkeypatch.setattr(mac_prove, "store_for", lambda model: str(tmp_path / "store"))
    seen = {}

    def run(a, ctx):
        seen["action"] = a
        return MacProveOutput(
            model=a.model, steps=a.steps, seq_len=a.seq_len, simulated=False,
            streamed={"block_size": 1, "peak_vram_gib": None, "s_per_step": 1, "final_loss": 1},
            resident={"block_size": 1, "peak_vram_gib": None, "s_per_step": 1, "final_loss": 1},
            identical=True, compared=[], receipt="r", env={}, method="ram-vs-ssd", witness={},
            machine="m",
        )

    monkeypatch.setattr(mac_prove, "run", run)
    yield tmp_path, seen
    contract.reset()


def _say(argv, tmp_path, capsys):
    ctx = Context(cwd=tmp_path, fixed_gpu=M4, fixed_cached=frozenset(table_models()),
                  fixed_interactive=False)
    code = app.dispatch(argv, ctx)
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    return code, json.loads(lines[-1])


def table_models():
    return ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-72B")


def test_no_arguments_is_the_small_model_at_its_table_length(words, capsys):
    tmp_path, seen = words
    code, obj = _say(["prove"], tmp_path, capsys)
    assert code == 0 and obj["method"] == "ram-vs-ssd", obj
    assert (seen["action"].model, seen["action"].seq_len) == ("Qwen/Qwen2.5-0.5B", 256)
    assert seen["action"].steps == 20


def test_a_large_model_takes_its_table_length_and_asks_in_terms_of_time(words, capsys):
    tmp_path, seen = words
    code, obj = _say(["prove", "Qwen/Qwen2.5-72B"], tmp_path, capsys)
    assert code == 2 and "twice" in obj["questions"][0]["prompt"]
    assert not BANNED.search(obj["questions"][0]["prompt"])
    code, _ = _say(["prove", "Qwen/Qwen2.5-72B", "--force"], tmp_path, capsys)
    assert code == 0 and seen["action"].seq_len == 128


def test_a_model_this_mac_does_not_train_is_refused(words, capsys):
    tmp_path, _ = words
    code, obj = _say(["prove", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"], tmp_path, capsys)
    assert code == 4 and obj["error"]["code"] == "E_GUARANTEE_UNAVAILABLE"


def test_one_step_is_raised_to_two(words, capsys):
    tmp_path, seen = words
    assert _say(["prove", "--steps", "1"], tmp_path, capsys)[0] == 0
    assert seen["action"].steps == 2


@pytest.mark.skipif(sys.platform == "darwin", reason="a Mac runs its own, for real")
def test_linux_runs_its_own_prove(tmp_path, monkeypatch):
    monkeypatch.setattr("poros.cli.verbs.prove_run.run", lambda a, ctx: "linux")
    from poros.cli.verbs.prove import run

    assert run(ProveAction(), Context(cwd=tmp_path)) == "linux"
