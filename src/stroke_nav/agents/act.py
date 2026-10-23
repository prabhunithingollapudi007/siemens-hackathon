"""Action-chunking policy (ACT-style) for the organiser observation, trained from demonstrations.

Behaviour cloning one action at a time (``scripts/train_bc.py``) learns a reactive mapping that
compounds errors. Action chunking (Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation
with Low-Cost Hardware") predicts the next ``chunk`` actions at once with a transformer decoder and
executes them with temporal ensembling, which smooths the roll-then-push phases of the entry-roll
expert. This is the deterministic core of ACT without its CVAE prior: one goal per case leaves no
multimodality worth modelling, and the smaller model fits a minutes-long training budget.

Inputs are the organiser observation (``image`` (1, 128, 128) uint8, ``state`` (43,) float32);
outputs are chunks of actions in [-1, 1] in the organiser order ``[v_gw, w_gw, v_cath, w_cath]``.
The model is saved as a plain ``torch`` checkpoint (``save_act`` / ``load_act``); it is **not** an
SB3 SAC model and does not load in the organiser's RL evaluation. Its product is the trajectory it
produces, which ``scripts/plan_case.py`` records in the common candidate format.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

STATE_DIM, ACTION_DIM = 43, 4


@dataclass
class ACTConfig:
    chunk: int = 10  # actions predicted per observation (2 s at 5 Hz)
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    use_image: bool = True
    lr: float = 3e-4
    batch_size: int = 64
    state_noise: float = 0.002  # Gaussian augmentation on the state (observation units)
    ensemble_m: float = 0.1  # temporal ensembling weight exp(-m * age), as in ACT
    seed: int = 0


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def build_model(cfg: ACTConfig):
    torch, nn = _torch()

    class ImageEncoder(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),  # 64
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  # 8
                nn.AdaptiveAvgPool2d(2))  # 4 spatial tokens
            self.proj = nn.Linear(64, d)

        def forward(self, img):  # (B, 1, 128, 128) in [0, 1] -> (B, 4, d)
            f = self.conv(img)
            return self.proj(f.flatten(2).transpose(1, 2))

    class ACTPolicy(nn.Module):
        def __init__(self, cfg: ACTConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.d_model
            self.state_proj = nn.Sequential(nn.Linear(STATE_DIM, d), nn.ReLU(), nn.Linear(d, d))
            self.image = ImageEncoder(d) if cfg.use_image else None
            n_mem = 1 + (4 if cfg.use_image else 0)
            self.mem_pos = nn.Parameter(torch.zeros(1, n_mem, d))
            self.queries = nn.Parameter(torch.zeros(1, cfg.chunk, d))
            nn.init.normal_(self.mem_pos, std=0.02)
            nn.init.normal_(self.queries, std=0.02)
            layer = nn.TransformerDecoderLayer(d, cfg.n_heads, dim_feedforward=4 * d, dropout=cfg.dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(layer, cfg.n_layers)
            self.head = nn.Linear(d, ACTION_DIM)

        def forward(self, image, state):  # image (B,1,128,128) float in [0,1] or None; state (B,43)
            tokens = [self.state_proj(state).unsqueeze(1)]
            if self.image is not None:
                tokens.append(self.image(image))
            memory = torch.cat(tokens, dim=1) + self.mem_pos
            out = self.decoder(self.queries.expand(state.shape[0], -1, -1), memory)
            return torch.tanh(self.head(out))  # (B, chunk, 4)

    return ACTPolicy(cfg)


# ---------------------------------------------------------------------- data
def make_chunks(episodes: list[dict], chunk: int) -> dict[str, np.ndarray]:
    """Training samples from demonstrations (``images``, ``states``, ``labels`` per episode): for each
    step the next ``chunk`` expert labels, padded with the last label where the episode ends."""
    images, states, targets = [], [], []
    for ep in episodes:
        lab = np.asarray(ep["labels"], np.float32)
        n = len(lab)
        if n == 0:
            continue
        pad = np.concatenate([lab, np.repeat(lab[-1:], chunk, axis=0)])
        images.append(np.asarray(ep["images"], np.uint8))
        states.append(np.asarray(ep["states"], np.float32))
        targets.append(np.stack([pad[t:t + chunk] for t in range(n)]))
    if not images:
        raise ValueError("no demonstration steps")
    return {"images": np.concatenate(images), "states": np.concatenate(states), "targets": np.concatenate(targets)}


def train_act(episodes: list[dict], cfg: ACTConfig = ACTConfig(), epochs: int = 50, max_seconds: float | None = None,
              val_fraction: float = 0.15, device: str = "auto", log=print):
    """Fit ``ACTPolicy`` to demonstrations (L1 on action chunks, as ACT). Stops after ``epochs`` or when
    ``max_seconds`` of wall time have passed, whichever comes first. Returns (model, history)."""
    torch, _ = _torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(val_fraction * len(episodes)))) if len(episodes) > 1 else 0
    val = make_chunks([episodes[i] for i in order[:n_val]], cfg.chunk) if n_val else None
    train = make_chunks([episodes[i] for i in order[n_val:]], cfg.chunk)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    def batch(data, idx, augment):
        st = data["states"][idx]
        if augment and cfg.state_noise > 0:
            st = st + rng.normal(0.0, cfg.state_noise, st.shape).astype(np.float32)
        img = torch.as_tensor(data["images"][idx], device=device).float().div_(255.0) if cfg.use_image else None
        return img, torch.as_tensor(np.clip(st, -10.0, 10.0), device=device), torch.as_tensor(data["targets"][idx], device=device)

    history, t0, n = [], time.monotonic(), len(train["targets"])
    for epoch in range(epochs):
        model.train()
        perm, losses = rng.permutation(n), []
        for s in range(0, n, cfg.batch_size):
            img, st, tgt = batch(train, perm[s:s + cfg.batch_size], True)
            loss = torch.nn.functional.l1_loss(model(img, st), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": epoch + 1, "train_l1": float(np.mean(losses)), "seconds": round(time.monotonic() - t0, 1)}
        if val is not None:
            model.eval()
            with torch.no_grad():
                vl = [torch.nn.functional.l1_loss(model(*batch(val, np.arange(s, min(len(val["targets"]), s + 256)), False)[:2]),
                                                  batch(val, np.arange(s, min(len(val["targets"]), s + 256)), False)[2]).item()
                      for s in range(0, len(val["targets"]), 256)]
            row["val_l1"] = float(np.mean(vl))
        history.append(row)
        log(f"[act] epoch {epoch + 1:3d}  train L1 {row['train_l1']:.4f}" + (f"  val L1 {row['val_l1']:.4f}" if val is not None else "")
            + f"  {row['seconds']:.0f} s")
        if max_seconds is not None and time.monotonic() - t0 >= max_seconds:
            log(f"[act] time budget of {max_seconds:.0f} s reached after {epoch + 1} epochs")
            break
    model.eval()
    return model, history


def save_act(model, cfg: ACTConfig, path: str | Path) -> Path:
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(cfg), "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, path)
    return path


def load_act(path: str | Path, device: str = "cpu"):
    torch, _ = _torch()
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = ACTConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


# ---------------------------------------------------------------------- execution
class ChunkExecutor:
    """Runs an ``ACTPolicy`` step by step with temporal ensembling: every step predicts a new chunk,
    and the executed action is the exp(-m * age)-weighted mean of all predictions for this step."""

    def __init__(self, model, cfg: ACTConfig, device: str = "cpu"):
        self.model, self.cfg, self.device = model, cfg, device
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.pending: list[tuple[int, np.ndarray]] = []  # (step predicted at, chunk)

    def act(self, obs: dict) -> np.ndarray:
        torch, _ = _torch()
        with torch.no_grad():
            img = torch.as_tensor(np.asarray(obs["image"], np.uint8)[None], device=self.device).float().div_(255.0) \
                if self.cfg.use_image else None
            st = torch.as_tensor(np.clip(np.asarray(obs["state"], np.float32), -10.0, 10.0)[None], device=self.device)
            chunk = self.model(img, st)[0].cpu().numpy()
        self.pending.append((self.t, chunk))
        self.pending = [(t0, c) for t0, c in self.pending if self.t - t0 < self.cfg.chunk]
        acts = np.stack([c[self.t - t0] for t0, c in self.pending])
        w = np.exp(-self.cfg.ensemble_m * np.array([self.t - t0 for t0, _ in self.pending], float))
        self.t += 1
        return np.clip((acts * w[:, None]).sum(0) / w.sum(), -1.0, 1.0).astype(np.float32)
