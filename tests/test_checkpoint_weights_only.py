"""Checkpoint artifacts must load under ``torch.load(weights_only=True)``.

Poros loads adapter weights, optimizer state, LR-scheduler state, and RNG
state with ``weights_only=True`` (pickle-safe) on resume. This guards against
a regression where a checkpoint artifact contains a type the safe unpickler
rejects, which would silently break ``--resume-from``. It also documents that
the hardening is intentional, not accidental.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
optim = torch.optim
LambdaLR = torch.optim.lr_scheduler.LambdaLR


def _make_checkpoint(tmp_path):
    """Write the four artifact files exactly as the trainer's saver does."""
    model = nn.Linear(8, 8)
    opt = optim.AdamW(model.parameters(), lr=1e-4)
    # one step to populate Adam moment buffers (exp_avg / exp_avg_sq)
    model(torch.randn(4, 8)).sum().backward()
    opt.step()
    sched = LambdaLR(opt, lr_lambda=lambda step: 1.0)

    torch.save(model.state_dict(), tmp_path / "adapter_weights.pt")
    torch.save(opt.state_dict(), tmp_path / "optimizer_state.pt")
    torch.save(sched.state_dict(), tmp_path / "lr_scheduler_state.pt")
    torch.save({"torch_rng": torch.get_rng_state()}, tmp_path / "rng_state.pt")
    return model, opt, sched


@pytest.mark.parametrize(
    "name",
    ["adapter_weights.pt", "optimizer_state.pt", "lr_scheduler_state.pt", "rng_state.pt"],
)
def test_checkpoint_artifact_loads_weights_only(tmp_path, name):
    _make_checkpoint(tmp_path)
    # The exact call pattern Poros uses on resume; must not raise under the
    # pickle-safe loader.
    state = torch.load(tmp_path / name, map_location="cpu", weights_only=True)
    assert state is not None


def test_optimizer_round_trips_under_weights_only(tmp_path):
    model, opt, _ = _make_checkpoint(tmp_path)
    fresh = optim.AdamW(model.parameters(), lr=1e-4)
    fresh.load_state_dict(
        torch.load(tmp_path / "optimizer_state.pt", map_location="cpu", weights_only=True)
    )
    # Adam moment buffers survived the safe load.
    assert any(
        "exp_avg" in s for s in fresh.state_dict()["state"].get(0, {})
    ) or fresh.state_dict()["state"], "optimizer state did not restore"
