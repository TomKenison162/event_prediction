"""Insider-flow model: does the trade tape know something the price doesn't?

Transformer over the last SEQ trades before a snapshot (each trade = token with
size, direction, price, timing and point-in-time wallet reputation) + market
context -> three heads:
  soon : market resolves YES within 72h (the event happens)
  yes  : market eventually resolves YES
  up   : YES price rises >= 10 points in the next 24h
The current price is an input, so any gain over the price-only baseline is
information carried by *who* is trading *how*.

Split by market close time: train < 2025-10-01 <= val < 2026-04-01 <= test.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from ..config import MODELS
from ..train_seq import device
from .data import POLY, load_trades
from .features import CTX_FEATS, SEQ, TOK_FEATS, build_tapes, context, enrich_trades, make_samples, window

HEADS = ["soon", "yes", "up"]
VAL_START, TEST_START = "2025-10-01", "2026-04-01"


class InsiderNet(nn.Module):
    def __init__(self, n_tok=len(TOK_FEATS), n_ctx=len(CTX_FEATS), d=128, heads=4, layers=4, dropout=0.1):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(n_tok, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout, batch_first=True, norm_first=True), layers)
        self.ctx = nn.Sequential(nn.Linear(n_ctx, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Linear(d, len(HEADS)))

    def forward(self, tok, mask, ctx):
        h = self.tok(tok)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
        m = torch.cat([torch.zeros(len(mask), 1, dtype=torch.bool, device=mask.device), mask], 1)
        h = self.enc(h, src_key_padding_mask=m)[:, 0]
        return self.head(torch.cat([h, self.ctx(ctx)], -1))


class Batches:
    """Builds normalised (tok, mask, ctx) batches on the fly from the tapes."""

    def __init__(self, tapes, S: pd.DataFrame, stats=None, seed=0):
        self.tapes, self.S = tapes, S.reset_index(drop=True)
        self._cid = self.S["cid"].to_numpy()
        self._i = self.S["i"].to_numpy()
        self._tau = self.S["tau"].to_numpy()
        self.ctx = np.stack([context(tapes[r.cid], r.i, r.tau, r.end, r.created)
                             for r in self.S.itertuples()]) if len(S) else np.zeros((0, len(CTX_FEATS)))
        if stats is None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(self.S), min(20_000, len(self.S)), replace=False)
            tok, mask = self._raw(idx)
            v = tok[~mask]
            stats = {"tmu": v.mean(0).tolist(), "tsd": (v.std(0) + 1e-6).tolist(),
                     "cmu": self.ctx.mean(0).tolist(), "csd": (self.ctx.std(0) + 1e-6).tolist()}
        self.stats = stats
        self._tmu, self._tsd = np.array(stats["tmu"], np.float32), np.array(stats["tsd"], np.float32)
        c = (self.ctx - np.array(stats["cmu"], np.float32)) / np.array(stats["csd"], np.float32)
        self.ctxn = np.clip(c, -8, 8).astype(np.float32)

    def _raw(self, idx):
        tok = np.zeros((len(idx), SEQ, len(TOK_FEATS)), np.float32)
        mask = np.zeros((len(idx), SEQ), bool)
        for k, j in enumerate(idx):
            tok[k], mask[k] = window(self.tapes[self._cid[j]], int(self._i[j]), int(self._tau[j]))
        return tok, mask

    def get(self, idx):
        tok, mask = self._raw(idx)
        tok = np.clip((tok - self._tmu) / self._tsd, -8, 8)
        tok[mask] = 0
        return tok, mask, self.ctxn[idx]

    def __len__(self):
        return len(self.S)


def prepare(rebuild: bool = False):
    cache = POLY / "insider_samples.pkl"
    if cache.exists() and not rebuild:
        return pickle.load(open(cache, "rb"))
    markets = pd.read_parquet(POLY / "markets_resolved.parquet")
    t0 = time.time()
    tr = load_trades(markets["condition_id"])
    print(f"{len(tr):,} trades from {tr['condition_id'].nunique()} markets ({time.time() - t0:.0f}s)")
    enr = enrich_trades(tr, markets)
    tapes = build_tapes(enr)
    S = make_samples(tapes, markets)
    S["close_dt"] = pd.to_datetime(S["close"], unit="s", utc=True)
    assert (S["tau"] < S["close"]).all(), "snapshot after close"
    print(f"{len(S):,} samples; soon {S.y_soon.mean():.3f} yes {S.y_yes.mean():.3f} up {S.y_up.mean():.3f}")
    out = (tapes, S)
    pickle.dump(out, open(cache, "wb"))
    return out


def predict(model, B: Batches, dev, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(B), bs):
            t, m, c = B.get(np.arange(i, min(i + bs, len(B))))
            out.append(torch.sigmoid(model(torch.tensor(t, device=dev), torch.tensor(m, device=dev),
                                           torch.tensor(c, device=dev))).float().cpu().numpy())
    return np.concatenate(out)


def evaluate(name, y, p) -> dict:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return {"model": name, "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--final", action="store_true", help="train on everything (fixed epochs)")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = device()
    tapes, S = prepare(a.rebuild)
    tr = S[S.close_dt < VAL_START] if not a.final else S
    va = S[(S.close_dt >= VAL_START) & (S.close_dt < TEST_START)]
    te = S[S.close_dt >= TEST_START]
    print(f"train {len(tr):,} val {len(va):,} test {len(te):,} samples; device {dev}")

    Btr = Batches(tapes, tr)
    stats = Btr.stats
    Bva, Bte = Batches(tapes, va, stats), Batches(tapes, te, stats)
    ytr = tr[["y_soon", "y_yes", "y_up"]].to_numpy(np.float32)

    # baselines: (1) price only, (2) logistic regression on market context (no tape)
    report = {}
    for h, col in zip(HEADS, ["y_soon", "y_yes", "y_up"]):
        lr = LogisticRegression(max_iter=2000).fit(Btr.ctxn, tr[col])
        rows = [evaluate("context_logreg", te[col], lr.predict_proba(Bte.ctxn)[:, 1])]
        if h == "yes":
            rows.insert(0, evaluate("price_only", te[col], te["p_now"].clip(0.001, 0.999)))
        report[h] = rows

    model = InsiderNet(d=a.d, heads=a.heads, layers=a.layers).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    pos = ytr.mean(0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.clip(0.3 / np.maximum(pos, 1e-3), 1, 10),
                                                         dtype=torch.float32, device=dev))
    best, bad, best_state = 1e9, 0, None
    for ep in range(a.epochs):
        model.train()
        perm = np.random.permutation(len(ytr))
        t0, tot = time.time(), 0.0
        for i in range(0, len(perm), a.bs):
            b = perm[i:i + a.bs]
            t, m, c = Btr.get(b)
            out = model(torch.tensor(t, device=dev), torch.tensor(m, device=dev), torch.tensor(c, device=dev))
            loss = lossf(out, torch.tensor(ytr[b], device=dev))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        msg = f"ep {ep:02d} train {tot / len(ytr):.4f} ({time.time() - t0:.0f}s)"
        if not a.final:
            pv = predict(model, Bva, dev)
            vl = float(np.mean([log_loss(va[c], np.clip(pv[:, k], 1e-5, 1 - 1e-5), labels=[0, 1])
                                for k, c in enumerate(["y_soon", "y_yes", "y_up"])]))
            msg += f" val_logloss {vl:.4f}"
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        print(msg, flush=True)
        if not a.final and bad >= a.patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    MODELS.mkdir(exist_ok=True)
    tag = "insider_final" if a.final else "insider"
    cfg = dict(d=a.d, heads=a.heads, layers=a.layers)
    torch.save({"cfg": cfg, "state": model.state_dict(), "stats": stats}, MODELS / f"{tag}.pt")
    if a.final:
        return

    # calibrate each head on val; report on test against baselines
    pv, pt = predict(model, Bva, dev), predict(model, Bte, dev)
    isos = {}
    for k, (h, col) in enumerate(zip(HEADS, ["y_soon", "y_yes", "y_up"])):
        isos[h] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(pv[:, k], va[col])
        report[h].append(evaluate("insider_transformer", te[col], isos[h].predict(pt[:, k])))
    pickle.dump(isos, open(MODELS / "insider_iso.pkl", "wb"))
    report["epochs_trained"] = ep + 1
    report["n"] = {"train": len(tr), "val": len(va), "test": len(te)}
    (MODELS / "insider_meta.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
