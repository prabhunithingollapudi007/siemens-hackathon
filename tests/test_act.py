"""ACT-style chunk policy: data chunking, a one-epoch fit, temporal-ensemble execution, save/load."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from stroke_nav.agents.act import ACTConfig, ChunkExecutor, load_act, make_chunks, save_act, train_act  # noqa: E402

CFG = ACTConfig(chunk=4, d_model=32, n_heads=2, n_layers=1, batch_size=8, lr=1e-3)


def _episode(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.zeros((n, 4), np.float32)
    labels[:, 0] = np.linspace(-1.0, 1.0, n)
    return {"images": rng.integers(0, 255, (n, 1, 128, 128), dtype=np.uint8),
            "states": rng.normal(0.0, 0.1, (n, 43)).astype(np.float32), "labels": labels}


def test_make_chunks_pads_with_the_last_label():
    ep = _episode(6, 0)
    d = make_chunks([ep], chunk=4)
    assert d["targets"].shape == (6, 4, 4) and d["images"].shape == (6, 1, 128, 128)
    assert np.allclose(d["targets"][0], ep["labels"][0:4])
    assert np.allclose(d["targets"][5], np.repeat(ep["labels"][5:6], 4, axis=0))
    assert np.allclose(d["targets"][4, 1:], np.repeat(ep["labels"][5:6], 3, axis=0))


def test_train_execute_save_load(tmp_path):
    episodes = [_episode(10, s) for s in range(3)]
    model, history = train_act(episodes, CFG, epochs=2, device="cpu", log=lambda *_: None)
    assert len(history) == 2 and "val_l1" in history[0]
    ex = ChunkExecutor(model, CFG)
    obs = {"image": episodes[0]["images"][0], "state": episodes[0]["states"][0]}
    a = ex.act(obs)
    assert a.shape == (4,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)
    for _ in range(6):  # more pending chunks than the chunk length: old ones are dropped
        ex.act(obs)
    assert len(ex.pending) == CFG.chunk

    f = save_act(model, CFG, tmp_path / "act.pt")
    model2, cfg2 = load_act(f)
    assert cfg2 == CFG
    img = torch.as_tensor(obs["image"][None]).float() / 255.0
    st = torch.as_tensor(obs["state"][None])
    with torch.no_grad():
        assert torch.allclose(model(img, st), model2(img, st))


def test_time_budget_stops_early():
    episodes = [_episode(8, s) for s in range(2)]
    _, history = train_act(episodes, CFG, epochs=50, max_seconds=0.0, device="cpu", log=lambda *_: None)
    assert len(history) == 1


def test_state_only_model():
    cfg = ACTConfig(chunk=3, d_model=16, n_heads=2, n_layers=1, batch_size=4, use_image=False)
    model, _ = train_act([_episode(6, 1), _episode(6, 2)], cfg, epochs=1, device="cpu", log=lambda *_: None)
    ex = ChunkExecutor(model, cfg)
    assert ex.act({"image": np.zeros((1, 128, 128), np.uint8), "state": np.zeros(43, np.float32)}).shape == (4,)
