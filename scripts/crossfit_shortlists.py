#!/usr/bin/env python3
"""
Cross-fitted shortlists for CHARM: every training conversation is ranked by a
retriever that never trained on it.

A retriever ranks its own training conversations far too well (target in its
top-200: 97.6% vs 60% on test), so a re-ranker trained on those shortlists sees
easy negatives it will never meet at test. Here the training conversations are
split into --cv-folds folds (run_experiment.py --cv-folds N --cv-fold k trains a
retriever without fold k); fold k's shortlists come from that retriever. The
validation and test shortlists come from the deployed retriever, exactly as the
agents see them. Cold-start item vectors (coldstart_probe.py's chosen config)
are applied to every retriever, with counts from its own training rows.

Writes a cache in train_charm_ce.py's shortlist format (--shortlists).

    python scripts/crossfit_shortlists.py --folds .../fold0/checkpoints/sft_final,.../fold1/... \\
        --deployed .../7b_val/checkpoints/sft_final --coldstart .../coldstart_probe.json \\
        --val-fraction 0.05 --charm-fraction 0.0 --top-k 200 --out shortlists_cv.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", required=True,
                        help="comma-separated checkpoints; the k-th trained without fold k")
    parser.add_argument("--deployed", required=True, help="checkpoint for validation and test")
    parser.add_argument("--coldstart", default=None, help="coldstart_probe.py output")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--limit-rows", type=int, default=0, help="smoke tests only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from run_experiment import (build_eval_catalog, build_model, coldstart_parts, cv_fold,
                                expand_mentions, load_split, split_conversations)
    from train_charm_ce import build_rows, compute_shortlists
    from harpo.coldstart import ColdStart, cold_items, training_counts
    from harpo.training import HARPOMTv2Trainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    folds = args.folds.split(",")
    with open(os.path.join(args.deployed, "..", "..", "catalog.json")) as f:
        catalog_list = json.load(f)
    for ck in folds:
        with open(os.path.join(ck, "..", "..", "catalog.json")) as f:
            if json.load(f) != catalog_list:
                raise ValueError(f"{ck} uses a different catalogue order")
    index = {t.lower(): i for i, t in enumerate(catalog_list)}
    rows = build_rows(args.data, index, args.val_fraction, args.charm_fraction, args.seed,
                      args.limit_rows)
    cfg = None
    if args.coldstart:
        with open(args.coldstart) as f:
            cfg = ColdStart(**json.load(f)["chosen"])
    profiles = {}
    if cfg is not None and cfg.profiles:
        with open(args.profiles) as f:
            profiles = json.load(f)
    train_raw, _, charm_raw = split_conversations(
        load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
        args.val_fraction, args.charm_fraction)
    backbone_rows = expand_mentions(train_raw)       # what the deployed retriever trained on

    def lists_for(ck, row_sets, count_rows):
        model, training_config = build_model(os.path.join(ck, "base_model"), device,
                                             args.seq_len, 16, False)
        HARPOMTv2Trainer(model, training_config, device=device).load_checkpoint(ck)
        model.eval()
        _, catalog = build_eval_catalog(model, catalog_list, device)
        if cfg is not None:
            counts = training_counts(count_rows, index, len(catalog_list))
            vecs, bias = cold_items(coldstart_parts(model, catalog_list, device, profiles),
                                    counts, cfg)
            catalog.override(vecs, bias)
        out = [compute_shortlists(model, catalog, rs, index, device, args.top_k, args.seq_len)
               if rs else None for rs in row_sets]
        del model, catalog
        if device == "cuda":
            torch.cuda.empty_cache()
        return out

    n = len(folds)
    fold_of = [cv_fold(r, n) for r in rows["train"]]
    parts = {}
    for k, ck in enumerate(folds):
        members = [i for i, f in enumerate(fold_of) if f == k]
        if not members:
            print(f"fold {k}: no training cases (tiny --limit-rows?)", flush=True)
            continue
        held = [rows["train"][i] for i in members]
        # The fold-k retriever trained on the other folds (of the non-validation conversations).
        seen = [r for r in backbone_rows if cv_fold(r, n) != k]
        (lists,) = lists_for(ck, [held], seen)
        parts[k] = (members, lists)
        print(f"fold {k}: {len(held)} training cases, target in top-{args.top_k}: "
              f"{100 * float((lists['target_pos'] >= 0).float().mean()):.1f}%", flush=True)

    train = {}
    total = len(rows["train"])
    first_part = next(iter(parts.values()))[1]
    for key in first_part:
        like = first_part[key]
        full = torch.zeros((total,) + tuple(like.shape[1:]), dtype=like.dtype)
        for members, lists in parts.values():
            full[torch.tensor(members, dtype=torch.long)] = lists[key]
        train[key] = full
    val, test = lists_for(args.deployed, [rows["val"], rows["test"]], backbone_rows)
    for name, v in (("train (cross-fitted)", train), ("val", val), ("test", test)):
        print(f"  {name}: target in top-{args.top_k}: "
              f"{100 * float((v['target_pos'] >= 0).float().mean()):.1f}%", flush=True)
    torch.save({"train": train, "val": val, "test": test,
                "meta": {"folds": folds, "deployed": args.deployed,
                         "coldstart": cfg.to_dict() if cfg else None}}, args.out + ".partial")
    os.replace(args.out + ".partial", args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
