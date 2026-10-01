#!/usr/bin/env python3
"""
MAVEN with its per-dialogue gate regularised, the strength chosen by
cross-validation on the validation set only.

The gate (a linear map from agent agreement/confidence statistics to
per-dialogue agent weights) was fitted on ~1.4k validation cases with almost no
regularisation, and on unseen data it did worse than one static weighting. Here
its weights carry an L2 penalty beta (beta -> infinity is the static
consensus). The candidates are fixed in advance -- static, then beta in
{10, 1, 0.1, 0.01, 0}, simplest first -- and compared by K-fold
cross-validation over validation *conversations*, repeated with different fold
assignments; the simplest candidate within one standard error of the best is
kept. It is refitted on all of validation, CHARM's diversity strength is chosen
on validation as before, and test is evaluated once.

    python scripts/maven_cv.py --test-raw .../rerank_top200_raw.pt --val-raw .../rerank_top200_val_raw.pt \\
        --test-ce charm_v2p_test.pt --val-ce charm_v2p_val.pt --agent star=star_test.pt,star_val.pt \\
        --agent charm_v1=.../charm_test.pt,.../charm_val.pt --diversity-profiles profiles.json --out maven_cv.json
"""

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch

CANDIDATES = [("static", None), ("gate beta=10", 10.0), ("gate beta=1", 1.0),
              ("gate beta=0.1", 0.1), ("gate beta=0.01", 0.01), ("gate beta=0", 0.0)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-raw", required=True)
    parser.add_argument("--val-raw", required=True)
    parser.add_argument("--test-ce", required=True)
    parser.add_argument("--val-ce", required=True)
    parser.add_argument("--agent", action="append", default=[],
                        help="extra agent: name=test_scores.pt,val_scores.pt")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--diversity-profiles", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from harpo.maven import (MAVENConsensus, agreement_features, fit, load_agent_scores,
                             ranks_from_scores, zrow)
    from harpo.ranking import metrics_from_ranks

    # This is all small-tensor CPU work by construction (no .cuda() anywhere in
    # this module), which left an idle GPU doing nothing for a stage that, in
    # practice, took hours of wall-clock time. Move it to GPU when available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    def to_device(d):
        for key, v in d.items():
            if isinstance(v, torch.Tensor):
                d[key] = v.to(device)
            elif isinstance(v, dict):
                to_device(v)
        return d

    torch.manual_seed(0)
    extras = [(s.split("=", 1)[0], *s.split("=", 1)[1].split(",")) for s in args.agent]
    test = to_device(load_agent_scores(args.test_raw, args.test_ce, [(n, t) for n, t, _ in extras]))
    val = to_device(load_agent_scores(args.val_raw, args.val_ce, [(n, v) for n, _, v in extras]))
    names = list(test["agents"])
    conv = torch.load(args.val_raw)["conversation_ids"]

    def tensors(d):
        z = torch.stack([zrow(d["agents"][n]) for n in names], -1)
        return z, agreement_features(z)

    z_v, f_v = tensors(val)
    z_t, f_t = tensors(test)

    def fitted(beta, idx):
        model = MAVENConsensus(len(names), f_v.size(1), gated=beta is not None).to(device)
        fit(model, z_v[idx], f_v[idx], val["target_pos"][idx], epochs=args.epochs,
            gate_l2=beta or 0.0)
        return model

    def metrics(fused, d, idx=None):
        idx = torch.arange(fused.size(0), device=fused.device) if idx is None else idx
        r = ranks_from_scores(fused, d["target_pos"][idx], d["retr_rank"][idx])
        dd = d["dedup"][idx]
        return {"standard": metrics_from_ranks(r.tolist(), d["pool"], (1, 10, 50)),
                "dedup": metrics_from_ranks(r[dd].tolist(), d["pool"], (1, 10, 50))}

    # ---- cross-validation on validation conversations
    # Checkpointed per (rep, fold) group (6 candidates each): a state file next
    # to --out tracks which groups are already scored, so a killed/interrupted
    # run resumes instead of redoing all `repeats * folds` groups. Fold
    # assignment is a deterministic hash of (rep, conversation id), so it is
    # identical across runs and safe to resume into.
    rows = torch.arange(len(conv), device=device)
    state_path = args.out + ".cv_state.json"
    scores = {name: [] for name, _ in CANDIDATES}
    done = set()
    if os.path.exists(state_path):
        with open(state_path) as f:
            saved = json.load(f)
        if saved.get("repeats") == args.repeats and saved.get("folds") == args.folds:
            scores = saved["scores"]
            done = {tuple(x) for x in saved["done"]}
            print(f"resuming CV from {state_path}: {len(done)}/{args.repeats * args.folds} "
                  f"(rep, fold) groups already done", flush=True)
        else:
            print(f"ignoring {state_path}: --repeats/--folds differ from this run", flush=True)

    total_groups = args.repeats * args.folds
    for rep in range(args.repeats):
        fold = torch.tensor([int(hashlib.md5(f"maven-cv{rep}:{c}".encode()).hexdigest(), 16) % args.folds
                             for c in conv], device=device)
        for k in range(args.folds):
            if (rep, k) in done:
                continue
            train, held = rows[fold != k], rows[fold == k]
            for name, beta in CANDIDATES:
                model = fitted(beta, train)
                with torch.no_grad():
                    fused, _ = model(z_v[held], f_v[held])
                scores[name].append(metrics(fused, val, held)["standard"]["mrr"])
            done.add((rep, k))
            print(f"  CV group {len(done)}/{total_groups} (rep={rep} fold={k}) done", flush=True)
            with open(state_path, "w") as f:
                json.dump({"repeats": args.repeats, "folds": args.folds,
                           "scores": scores, "done": sorted(done)}, f)
    table = {}
    for name, _ in CANDIDATES:
        x = torch.tensor(scores[name])
        table[name] = {"mean": float(x.mean()), "se": float(x.std() / math.sqrt(len(x)))}
    best = max(table, key=lambda n: table[n]["mean"])
    bar = table[best]["mean"] - table[best]["se"]
    chosen = next(n for n, _ in CANDIDATES if table[n]["mean"] >= bar)
    print(f"agents: {names}; {args.repeats} x {args.folds}-fold CV over "
          f"{len(set(conv))} validation conversations", flush=True)
    for name, _ in CANDIDATES:
        mark = "  <- chosen" if name == chosen else ("  (best)" if name == best else "")
        print(f"  {name:16s} CV MRR {table[name]['mean']:.4f} +- {table[name]['se']:.4f}{mark}", flush=True)

    # ---- refit on all of validation, diversity on validation, test once
    beta = dict(CANDIDATES)[chosen]
    model = fitted(beta, rows)
    with torch.no_grad():
        fused_t, w_t = model(z_t, f_t)
        fused_v, _ = model(z_v, f_v)
    report = {"MAVEN (CV-regularised)": {"metrics": metrics(fused_t, test),
                                         "mean_weights": w_t.mean(0).tolist()}}
    if args.diversity_profiles:
        from harpo.diversity import intra_list_diversity, mmr_rerank, profile_vectors, uncertainty

        with open(args.diversity_profiles) as f:
            vecs = profile_vectors(test["catalog"], json.load(f)).to(device)
        u_v, u_t = uncertainty(fused_v), uncertainty(fused_t)
        lam = max((0.0, 0.05, 0.1, 0.2, 0.3, 0.5), key=lambda l: metrics(
            mmr_rerank(fused_v, val["top_idx"], vecs, l * u_v), val)["standard"]["mrr"])
        rescored = mmr_rerank(fused_t, test["top_idx"], vecs, lam * u_t)
        report["MAVEN (CV-regularised) + CHARM diversity"] = {
            "metrics": metrics(rescored, test), "lambda0": lam,
            "ild_top10": {"before": intra_list_diversity(test["top_idx"], fused_t, vecs),
                          "after": intra_list_diversity(test["top_idx"], rescored, vecs)}}
    for label, r in report.items():
        s_, d_ = r["metrics"]["standard"], r["metrics"]["dedup"]
        print(f"TEST {label:42s} R@1 {100 * s_['recall_at_1']:.2f}  R@10 {100 * s_['recall_at_10']:.2f}  "
              f"R@50 {100 * s_['recall_at_50']:.2f}  NDCG@10 {100 * s_['ndcg_at_10']:.2f}  "
              f"MRR@10 {100 * s_['mrr_at_10']:.2f}  MRR {s_['mrr']:.4f} | dedup R@1 "
              f"{100 * d_['recall_at_1']:.2f}  R@10 {100 * d_['recall_at_10']:.2f}", flush=True)
    print(f"  mean weights {dict(zip(names, [round(x, 3) for x in w_t.mean(0).tolist()]))}"
          + (f"; diversity lambda0 {report[label]['lambda0']}" if args.diversity_profiles else ""))
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "agents": names, "cv": table, "chosen": chosen,
                   "report": report}, f, indent=2)
    print(f"wrote {args.out}")
    if os.path.exists(state_path):
        os.remove(state_path)


if __name__ == "__main__":
    main()
