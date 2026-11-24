"""Spatio-temporal transformer over the country x day panel.

One sample = one cutoff day d. Input: F[d-L+1 .. d, all countries, all features].
  1. per-country temporal transformer over the L days
  2. cross-country transformer over the per-country summaries (spillovers)
  3. head -> logits[country, type, horizon]
Labels: Y[d+h, country, type] for h in HORIZONS.

Runs on CUDA (rented GPU), Apple MPS, or CPU. Scale with --d-model/--layers.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .config import HORIZONS, MODELS, SEQ_LEN, TYPE_NAMES
from .dataset import Panel, build_panel
from .train_lgbm import TRAIN_END, VAL_END

H = len(HORIZONS)
T = len(TYPE_NAMES)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class STTransformer(nn.Module):
    def __init__(self, n_feat: int, n_countries: int, d_model=128, heads=4,
                 t_layers=3, c_layers=2, dropout=0.1, seq_len=SEQ_LEN):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.cemb = nn.Embedding(n_countries, d_model)
        enc = lambda n: nn.TransformerEncoder(  # noqa: E731
            nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout,
                                       batch_first=True, norm_first=True), n)
        self.temporal = enc(t_layers)
        self.spatial = enc(c_layers) if c_layers > 0 else nn.Identity()
        self.head = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model),
                                  nn.GELU(), nn.Linear(d_model, T * H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L, K]
        B, C, L, K = x.shape
        cid = torch.arange(C, device=x.device)
        h = self.inp(x) + self.pos[:, :L].unsqueeze(1) + self.cemb(cid)[None, :, None, :]
        h = self.temporal(h.reshape(B * C, L, -1))
        last = h[:, -1].reshape(B, C, -1)
        pooled = h.mean(1).reshape(B, C, -1)
        s = self.spatial(last)
        out = self.head(torch.cat([s, pooled], -1))
        return out.reshape(B, C, T, H)


class Data:
    """Holds the normalised panel on device and slices windows on the fly."""

    def __init__(self, p: Panel, stats: dict | None, dev):
        F = p.F
        if stats is None:
            tr = F[np.asarray(p.days <= pd.Timestamp(TRAIN_END))]
            stats = {"mu": tr.reshape(-1, F.shape[-1]).mean(0).tolist(),
                     "sd": (tr.reshape(-1, F.shape[-1]).std(0) + 1e-6).tolist()}
        self.stats = stats
        Fn = (F - np.array(stats["mu"], np.float32)) / np.array(stats["sd"], np.float32)
        self.F = torch.tensor(np.clip(Fn, -10, 10), dtype=torch.float32, device=dev)
        Ypad = np.concatenate([p.Y, np.zeros((max(HORIZONS),) + p.Y.shape[1:], bool)])
        self.Y = torch.tensor(Ypad, dtype=torch.float32, device=dev)
        self.n_days = len(p.days)
        self.hz = torch.tensor(HORIZONS, device=dev)

    def batch(self, cut: torch.Tensor):
        win = cut[:, None] - torch.arange(SEQ_LEN - 1, -1, -1, device=cut.device)[None]  # B,L
        x = self.F[win].permute(0, 2, 1, 3)                    # B,C,L,K
        y = self.Y[cut[:, None] + self.hz[None]]               # B,H,C,T
        return x, y.permute(0, 2, 3, 1)                        # B,C,T,H


def cutoffs(p: Panel, lo: str, hi: str) -> np.ndarray:
    idx = np.where((p.days >= pd.Timestamp(lo)) & (p.days <= pd.Timestamp(hi)))[0]
    return idx[(idx >= SEQ_LEN - 1) & (idx + max(HORIZONS) < len(p.days))]


@torch.no_grad()
def predict(model, data: Data, cut: np.ndarray, bs=32) -> np.ndarray:
    model.eval()
    out = []
    dev = data.F.device
    for i in range(0, len(cut), bs):
        x, _ = data.batch(torch.tensor(cut[i:i + bs], device=dev))
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)  # N,C,T,H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--t-layers", type=int, default=3)
    ap.add_argument("--c-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--final", action="store_true", help="train through latest data, fixed epochs")
    ap.add_argument("--tag", default="seq")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = device()
    p = build_panel()
    data = Data(p, None, dev)
    first = str(p.days[SEQ_LEN].date())
    last = str(p.days[-1].date())
    tr = cutoffs(p, first, last if a.final else TRAIN_END)
    va = cutoffs(p, "2024-07-01", VAL_END)
    te = cutoffs(p, "2025-07-01", last)
    print(f"device {dev}; panel {p.F.shape}; train {len(tr)} val {len(va)} test {len(te)} days")

    cfg = dict(n_feat=p.F.shape[-1], n_countries=len(p.countries), d_model=a.d_model,
               heads=a.heads, t_layers=a.t_layers, c_layers=a.c_layers, dropout=a.dropout)
    model = STTransformer(**cfg).to(dev)
    print(f"params: {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * (len(tr) // a.bs + 1))
    # up-weight positives a little (base rate is low) without distorting too much
    pos_rate = float(p.Y[tr].mean())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(5.0, 0.5 / max(pos_rate, 1e-3)), device=dev))

    best, bad, best_state = 1e9, 0, None
    for ep in range(a.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = np.random.permutation(tr)
        for i in range(0, len(perm), a.bs):
            x, y = data.batch(torch.tensor(perm[i:i + a.bs], device=dev))
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        msg = f"ep {ep:02d} train {tot / len(tr):.4f} ({time.time() - t0:.0f}s)"
        if not a.final:
            pv = predict(model, data, va)
            yv = data.Y[torch.tensor(va[:, None] + np.array(HORIZONS)[None], device=dev)].cpu().numpy()
            yv = yv.transpose(0, 2, 3, 1)
            vl = float(nn.functional.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6),
                                                          torch.tensor(yv)).item())
            msg += f" val_bce {vl:.4f}"
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg, flush=True)
        if not a.final and bad >= a.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    MODELS.mkdir(exist_ok=True)
    torch.save({"cfg": cfg, "state": model.state_dict(), "stats": data.stats}, MODELS / f"{a.tag}.pt")
    if a.final:
        print("final model saved")
        return

    # isotonic calibration on val, report on test
    pv = predict(model, data, va)
    yv = p.Y[va[:, None] + np.array(HORIZONS)[None]].transpose(0, 2, 3, 1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv.ravel(), yv.ravel())
    pickle.dump(iso, open(MODELS / f"{a.tag}_iso.pkl", "wb"))
    pt = predict(model, data, te)
    yt = p.Y[te[:, None] + np.array(HORIZONS)[None]].transpose(0, 2, 3, 1)
    rep = {"auc_test": float(roc_auc_score(yt.ravel(), pt.ravel())), "val_bce": best, "cfg": cfg,
           "epochs_trained": ep + 1}
    (MODELS / f"{a.tag}_meta.json").write_text(json.dumps(rep, indent=2))
    print(rep)


if __name__ == "__main__":
    main()
