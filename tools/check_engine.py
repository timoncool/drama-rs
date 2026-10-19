#!/usr/bin/env python3
"""Compare the Rust engine's per-step logits with tools/reference.py (teacher forced
on the engine's own sampled frames)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import reference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--binary", default="target/release/drama")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--work", default="test/work")
    ap.add_argument("--original", default=None, help="compare against the unquantized original checkpoint dir")
    ap.add_argument("--experts", default="q4", choices=["q4", "fp8"], help="engine routed experts (fp8 reads --original from host RAM)")
    ap.add_argument("--rows-file", default=None, help="use these prompt rows (raw i32 [rows][11]) instead of random ones")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)

    m = reference.Drama3Ref(args.model, original=args.original)
    if args.rows_file:
        frames = torch.from_numpy(np.fromfile(args.rows_file, dtype="<i4").astype(np.int64)).reshape(-1, 11)
    else:
        g = torch.Generator().manual_seed(0)
        L = args.rows
        frames = torch.zeros(L, 11, dtype=torch.int64)
        frames[:, 0] = torch.randint(0, 5000, (L,), generator=g)
        a, b = L // 3, 2 * L // 3
        frames[a:b, 1:] = torch.randint(0, m.cb, (b - a, 10), generator=g)
        frames[a:b, 0] = frames[a:b, 1] + m.sem0
    rows_path = os.path.join(args.work, "rows.bin")
    frames.numpy().astype("<i4").tofile(rows_path)

    trace_path = os.path.join(args.work, "trace.bin")
    cmd = [
        args.binary, "trace", "--model", args.model, "--rows", rows_path, "--steps", str(args.steps),
        "--out", trace_path, "--chunk", str(args.chunk), "--temperature", str(args.temperature), "--max-len", "1024",
    ]
    if args.experts == "fp8":
        cmd += ["--experts", "fp8", "--fp8-model", args.original]
    subprocess.run(cmd, check=True)

    n_cand = m.cb + 1
    nh = (m.ncb - 1) * m.cb
    rec = 11 + n_cand + nh
    raw = np.fromfile(trace_path, dtype="<i4")
    assert raw.size == rec * args.steps, (raw.size, rec * args.steps)
    raw = raw.reshape(args.steps, rec)
    eng_frames = raw[:, :11]
    eng_main = raw[:, 11 : 11 + n_cand].view("<f4")
    eng_heads = raw[:, 11 + n_cand :].view("<f4").reshape(args.steps, m.ncb - 1, m.cb)

    kls = []
    st = m.new_state()
    with torch.no_grad():
        hdn, pos, f = m.prefill(frames.cuda(), st)
        worst = 0.0
        for s in range(args.steps):
            tr = {}
            fr = [int(v) for v in eng_frames[s]]
            m.decode_frame(hdn, pos, f, st, 1.0, 30, 0.9, None, forced=fr, trace=tr)
            ref_main = tr["main_logits"].cpu().numpy()
            ref_heads = np.stack([tr[f"head{r}"].cpu().numpy() for r in range(m.ncb - 1)])

            def cmp(a, b):
                err = np.abs(a - b).max() / (np.abs(b).max() + 1e-9)
                cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                return err, cos

            def kl(a, b):
                """KL(ref || engine) of the softmax distributions, and top-1 agreement."""
                pa = torch.log_softmax(torch.from_numpy(a).double(), -1)
                pb = torch.log_softmax(torch.from_numpy(b).double(), -1)
                return float((pb.exp() * (pb - pa)).sum()), int(a.argmax() == b.argmax())

            kls.append([kl(eng_main[s], ref_main)] + [kl(eng_heads[s, r], ref_heads[r]) for r in range(m.ncb - 1)])

            e1, c1 = cmp(eng_main[s], ref_main)
            head_stats = [cmp(eng_heads[s, r], ref_heads[r]) for r in range(m.ncb - 1)]
            e2 = max(e for e, _ in head_stats)
            c2 = min(c for _, c in head_stats)
            worst = max(worst, e1, e2)
            print(f"step {s}: frame {fr[:4]}...  main rel-err {e1:.2e} cos {c1:.6f} | heads rel-err {e2:.2e} min-cos {c2:.6f}")
            hdn, pos, f = m.step(fr, st)
    print("worst relative error", worst)
    k = np.array(kls)  # [steps, 10, 2]
    print("mean KL(ref||engine) main %.4f, heads %s" % (k[:, 0, 0].mean(), np.round(k[:, 1:, 0].mean(0), 4).tolist()))
    print("top-1 agreement main %.2f, heads %s" % (k[:, 0, 1].mean(), np.round(k[:, 1:, 1].mean(0), 2).tolist()))


if __name__ == "__main__":
    main()
