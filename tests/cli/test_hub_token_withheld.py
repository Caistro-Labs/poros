"""A gated or private model an agent names is never refused on an answer the Hub gave a stranger.

Under `poros agents` no stored token is sent before the load for a name the catalog does not
hold (kept: an agent's words may be anyone's). But the check then read the Hub's tokenless
answer as the truth: a gated model the user is logged in with access to was refused before the
load (E_HUB_AUTH, "log in"), and a private one as "not on the Hub ... log in". Where the token
was withheld, "gated" now goes through to the load, which sends it; "not found" stays refused
and says why the token was not sent and what to do. On a stood-in Hub.
"""

from __future__ import annotations

import pytest

hub = pytest.importorskip("huggingface_hub")

from poros.preflight import command  # noqa: E402
from tests.cli.test_train_runs import home  # noqa: E402,F401  (the simulated machine)

CATALOG = "Qwen/Qwen2.5-0.5B-Instruct"
GATED, PRIVATE, NONE = "someone/gated-model", "someone/private-model", "someone/no-such-model"


def _error(kind: str):
    cls = getattr(hub.utils, kind)
    return cls.__new__(cls)  # its text is the library's: nothing here reads it


@pytest.fixture
def the_hub(monkeypatch, tmp_path):
    """A Hub where the user is logged in with access to every repo but NONE: a call with no
    token is answered as a stranger's. Returns the (model, token) of each call."""
    asked = []
    config = tmp_path / "config.json"
    config.write_text('{"model_type": "qwen2"}')

    def answer(model, token):
        asked.append((model, token))
        if model == NONE or model.startswith("Qwen/NoSuch"):
            raise _error("RepositoryNotFoundError")
        if token is False:  # no token: what anyone is told
            if model in (GATED, "Qwen/Gated-Catalog"):
                raise _error("GatedRepoError")
            if model == PRIVATE:
                raise _error("RepositoryNotFoundError")

    class Api:
        def model_info(self, model, revision=None, files_metadata=False, token=None):
            answer(model, token)
            return None

    def download(repo_id, filename, revision=None, token=None):
        answer(repo_id, token)
        return str(config)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.delenv("POROS_AGENT", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return asked


def _agent(monkeypatch):
    monkeypatch.setenv("POROS_AGENT", "1")


def test_a_person_with_access_passes_and_the_token_is_sent_only_when_needed(the_hub):
    for model in (GATED, PRIVATE):
        assert command.model_refusal(model) is None, model
        assert (model, False) in the_hub and (model, None) in the_hub  # none first, then theirs
    the_hub.clear()
    assert command.model_refusal(CATALOG) is None
    assert {token for _, token in the_hub} == {False}  # a public repo: no token at all
    gone = command.model_refusal(NONE)
    assert gone.code == "E_NOT_FOUND" and "hf auth login" in " ".join(gone.context)
    assert "poros agents" not in " ".join(gone.context)


def test_an_agents_gated_model_goes_through_to_the_load_with_no_token_sent(the_hub, monkeypatch):
    """Refused before: E_HUB_AUTH "is gated ... log in", to a user who is logged in with access."""
    _agent(monkeypatch)
    assert command.model_refusal(GATED) is None  # the load sends the token, and says
    assert the_hub and {token for _, token in the_hub} == {False}  # the stored token: never sent


def test_an_agents_missing_or_private_model_is_refused_and_says_why_no_token_was_sent(
        the_hub, monkeypatch):
    _agent(monkeypatch)
    for model in (PRIVATE, NONE):
        gone = command.model_refusal(model)
        assert (gone.code, gone.title) == ("E_NOT_FOUND", "No such model"), model
        said = " ".join(gone.context)
        assert "under poros agents no stored token is sent for a name the catalog does not" in said
        assert f"hf download {model} first, or name its folder" in said
        assert "hf auth login" not in said  # logged in already: not the next step
    assert {token for _, token in the_hub} == {False}


def test_an_agents_catalog_model_is_asked_with_the_token_and_refused_truly(the_hub, monkeypatch):
    """A catalog name gets the token, so its answers are this user's: a refusal there stands."""
    from poros import catalog

    _agent(monkeypatch)
    real = catalog.names
    monkeypatch.setattr(catalog, "names", lambda m: True if m.startswith("Qwen/") else real(m))
    assert command.model_refusal("Qwen/Gated-Catalog") is None  # gated, and they have access
    assert ("Qwen/Gated-Catalog", None) in the_hub
    gone = command.model_refusal("Qwen/NoSuch-Catalog")
    assert gone.code == "E_NOT_FOUND" and "hf auth login" in " ".join(gone.context)


def test_a_gated_model_with_no_access_is_still_refused_where_the_token_was_sent(the_hub,
                                                                               monkeypatch):
    def refuse_all(model, revision=None):
        raise _error("GatedRepoError")

    monkeypatch.setattr("poros.plan.configs.fetch_config", refuse_all)
    gone = command.model_refusal(GATED)  # a person: asked with their token, and still gated
    assert (gone.code, gone.title) == ("E_HUB_AUTH", "Needs access")


# ── the whole command, on the simulated trainer ────────────────────────────────────────────────


@pytest.mark.parametrize("agent", [False, True], ids=["person", "agent"])
def test_train_of_a_gated_model_runs_and_a_missing_one_is_refused(home, the_hub, monkeypatch,  # noqa: F811
                                                                  capsys, agent):
    """`poros agents train <gated id>` was refused before the load (E_HUB_AUTH, "log in") for
    a user logged in with access; it now trains, with no stored token sent before the load. A
    repo the Hub does not show stays refused, and says why under `poros agents`."""
    from tests.cli.test_train_runs import say

    if agent:
        _agent(monkeypatch)
    argv = ["--data", "notes.jsonl", "--steps", "2", "--yes", "--allow-unvalidated"]
    code, obj = say(["train", GATED, *argv], home, capsys)
    assert code == 0 and obj["schema"] == "poros.train.v1", obj
    tokens = {token for model, token in the_hub if model == GATED}
    assert tokens == ({False} if agent else {False, None})  # an agent's: never the stored one
    the_hub.clear()
    code, obj = say(["train", PRIVATE, *argv], home, capsys)
    if agent:
        assert code == 2 and obj["error"]["code"] == "E_NOT_FOUND", obj
        said = " ".join(obj["error"]["message"].split())
        assert "under poros agents no stored token is sent" in said
        assert f"hf download {PRIVATE} first, or name its folder" in said
        assert {token for _, token in the_hub} == {False}
    else:  # a person's call is asked with their token: the private repo they can read trains
        assert code == 0, obj
