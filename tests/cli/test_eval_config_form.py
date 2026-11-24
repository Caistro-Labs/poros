"""`poros eval <config>` after a config file's run says what the run trained to.

It scored every token of each row through the classic loop's evaluation, whatever trained the
adapter: an SFT run that ended at 0.0000 on its answers read `eval_loss 3.6493` on its own
training file, the loss over prompts and template it never trained on. A recipe's run is now
scored as it trained (the base, then the run, as `poros eval <run>`), and the classic loop's
number says what it is.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from poros.cli.schemas import EvalOutput, EvalSide


def _folder(tmp_path, name, **said):
    out = tmp_path / name
    out.mkdir(parents=True)
    if said:
        (out / "config.resolved.json").write_text(json.dumps(said))
    return out


def test_a_recipes_run_is_told_from_the_classic_loops(tmp_path):
    from poros.cli.commands.export import _recipe_run

    sft = _folder(tmp_path, "sft", trained_with="sft", data_shape="chats")
    dpo = _folder(tmp_path, "dpo", trained_with="dpo", data_shape="pairs")
    classic = _folder(tmp_path, "classic", trained_with="classic", data_shape=None)
    bare = _folder(tmp_path, "bare")
    ckpt = _folder(tmp_path, "sft/checkpoint-9")
    assert _recipe_run(sft, sft) == sft and _recipe_run(dpo, dpo) == dpo
    assert _recipe_run(ckpt, tmp_path / "elsewhere") == sft  # a checkpoint: its run's folder
    assert _recipe_run(classic, classic) is None
    assert _recipe_run(bare, bare) is None  # no record of how it trained: the old measure, named
    assert _recipe_run(bare, sft) == sft  # --adapter elsewhere: the config's own run says


def test_a_recipes_run_is_scored_as_it_trained_and_says_which_rows(tmp_path, monkeypatch,
                                                                    capsys):
    from poros.cli import contract
    from poros.cli.commands import export as command
    from poros.cli.verbs import evaluate as verb

    folder = _folder(tmp_path, "run", trained_with="sft", data_shape="chats")
    seen = {}

    def score(r, adapter, ctx, **how):
        seen.update(run=r, adapter=adapter, **how)
        return EvalOutput(adapter=str(adapter), dataset=str(tmp_path / "chats.jsonl"), run=r.id,
                          metrics={"eval_loss": 0.0002, "eval_perplexity": 1.0002},
                          shape="chats", rows_from="file", rows=20, of=20,
                          base=EvalSide(loss=2.1), trained=EvalSide(loss=0.0002), change=-0.9999,
                          improved=20)

    monkeypatch.setattr(verb, "score", score)
    monkeypatch.setenv("POROS_OUTPUT", "json")
    shown = []
    cfg = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
                          dataset=str(tmp_path / "chats.jsonl"))
    try:
        command._eval_as_trained(folder, folder, cfg, SimpleNamespace(print=shown.extend), 110)
    finally:
        contract.reset()
    assert seen["run"].output_dir == str(folder) and seen["dataset"] == cfg.dataset
    text = "\n".join("".join(t for t, _ in line) for line in shown)
    assert "Scored run" in text and "20 rows of chats.jsonl" in text
    assert "it held no rows out: these are rows it trained on" in text
    assert "2.100 base" in text and "0.000 trained" in text
    flat = " ".join(text.replace("\\\n", " ").split())  # a long command wraps on the screen
    assert f"poros export {folder} -o ./merged" in flat
    said = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert said["schema"] == "poros.eval.v1" and said["rows_from"] == "file"
    assert said["trained"]["loss"] == 0.0002 and said["base"]["loss"] == 2.1
    assert said.get("run") is None  # a config's run is no id among `poros runs`


def test_the_flag_reaches_a_recipe_runs_engine_and_no_file_grants_it(tmp_path, monkeypatch):
    """`poros eval run.yaml --trust-remote-code` on a recipe's run built its engine from the
    run's settings, which carry no such grant: the flag was ignored. Only the caller's flag
    grants it: a settings file that says so grants nothing."""
    from poros.cli.commands import export as command
    from poros.cli.context import Context
    from poros.cli.verbs import evaluate as verb
    from poros.cli.verbs import evaluate_pass as ep
    from poros.cli.verbs import shared
    from poros.core.config import PorosConfig

    folder = _folder(tmp_path, "run", trained_with="sft", data_shape="chats")
    passed = []
    monkeypatch.setattr(verb, "score", lambda r, adapter, ctx, **how: passed.append(how) or 1 / 0)
    cfg = SimpleNamespace(model_name_or_path="Org/Model", dataset="chats.jsonl")
    for flag in (True, False):
        with pytest.raises(BaseException):  # the stand-in stops there
            command._eval_as_trained(folder, folder, cfg, SimpleNamespace(print=list), 110,
                                     trust_remote_code=flag)
    assert [how["trust_remote_code"] for how in passed] == [True, False]

    class Reached(Exception):
        pass

    built = []

    def planted(model):
        raise Reached

    monkeypatch.undo()
    for said_by_file in (False, True):  # what the run's own record says: never the grant
        monkeypatch.setattr(ep, "settings", lambda r, adapter, f=said_by_file: (
            built.append(PorosConfig(model_name_or_path="Org/Model", trust_remote_code=f))
            or built[-1], "chats"))
        monkeypatch.setattr(shared, "refuse_planted", planted)
        for flag in (True, False):
            with pytest.raises(Reached):
                verb.score(verb.folder_run(folder, "Org/Model"), folder, Context(),
                           trust_remote_code=flag)
            assert built[-1].trust_remote_code is flag
        monkeypatch.undo()


def test_the_mismatch_note_says_which_settings_a_recipe_run_is_scored_with(tmp_path, capsys):
    """The note said "the eval config says ...; metrics from a mismatched base or LoRA shape
    are not meaningful" of a recipe's run too, whose engine is not built from the eval config."""
    from poros.cli.commands import export as command
    from poros.core.config import PorosConfig

    adapter = _folder(tmp_path, "adapter")
    (adapter / "poros_config.json").write_text(json.dumps({"lora_rank": 16}))
    cfg = PorosConfig(model_name_or_path="Org/Model", lora_rank=8)
    command._warn_on_adapter_config_mismatch(adapter, cfg, as_trained=adapter)
    said = "".join(capsys.readouterr().out.replace("│", " ").split())  # a long path wraps
    assert "trainedwith16,theevalconfigsays8" in said
    assert (f"scoredwiththesettings{adapter / 'config.resolved.json'}recorded,"
            "nottheevalconfig's") in said
    assert "notmeaningful" not in said
    command._warn_on_adapter_config_mismatch(adapter, cfg)  # the classic loop: as before
    assert "not meaningful" in " ".join(capsys.readouterr().out.split())


def test_the_datasets_librarys_bars_are_off_while_a_dataset_is_read():
    """`Generating train split`, `Map` and `Filter` drew three bars over the eval's screen."""
    pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    from poros.train.datasets import _no_progress_bars

    datasets.enable_progress_bars()
    with _no_progress_bars():
        assert not datasets.is_progress_bar_enabled()
    assert datasets.is_progress_bar_enabled()  # as they were, once it is read
    datasets.disable_progress_bars()
    with _no_progress_bars():
        pass
    assert not datasets.is_progress_bar_enabled()  # and left off where its user turned them off
    datasets.enable_progress_bars()
