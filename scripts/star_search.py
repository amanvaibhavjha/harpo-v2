#!/usr/bin/env python3
"""
STAR scores for a saved shortlist: tree search over preference readings with
CHARM as the value network (harpo/star_search.py).

Reads a rerank_eval.py raw dump and searches over the top --top-m candidates of
each distinct dialogue. Writes ``[N, K]`` scores aligned to the dump (the first
M columns are STAR's; the rest get the row mean, so STAR is neutral about
candidates it did not search), plus the root-only scores (CHARM under the greedy
reading, no search) so the search's own contribution can be measured.

    python scripts/star_search.py --lm Qwen/Qwen2.5-7B-Instruct --charm-adapter .../charm_ce_adapter \\
        --base-model Qwen/Qwen2.5-7B-Instruct --raw .../rerank_top200_raw.pt \\
        --summaries s0.json,s1.json --out star_test.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm", required=True, help="instruct model that proposes readings")
    parser.add_argument("--charm-adapter", required=True, help="CHARM adapter (trained with preference readings)")
    parser.add_argument("--base-model", required=True, help="CHARM's base model")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--summaries", required=True, help="root readings (pref_summaries.py)")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--charm-fraction", type=float, default=0.25)
    parser.add_argument("--top-m", type=int, default=30)
    parser.add_argument("--beam", type=int, default=3)
    parser.add_argument("--branch", type=int, default=2, help="children per node below depth 1")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--backtrack", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--dialogues-per-batch", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--limit", type=int, default=0, help="dialogues (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_experiment import expand_mentions, load_split, split_conversations
    from harpo.charm_ce import CrossEncoderCHARM, adapter_config
    from harpo.star import clean_reading, dialogue_key, load_readings, reading_messages
    from harpo.star_search import Node, aggregate, next_frontier, node_value, survivors

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype) if device == "cuda" else torch.float32
    raw = torch.load(args.raw)
    catalog = raw["catalog"]
    index = {t.lower(): i for i, t in enumerate(catalog)}
    if args.split == "test":
        rows = expand_mentions(load_split(os.path.join(args.data, "test_sft.json"), 0, args.seed))
    else:
        _, val_raw, _ = split_conversations(
            load_split(os.path.join(args.data, "sft_data.json"), 0, args.seed),
            args.val_fraction, args.charm_fraction)
        rows = expand_mentions(val_raw)
    rows = [r for r in rows if str(r["ground_truth_item"]).lower() in index]
    targets = torch.tensor([index[str(r["ground_truth_item"]).lower()] for r in rows])
    if not torch.equal(targets, raw["targets"].cpu()):
        raise ValueError("rows do not line up with the raw dump (different data or split)")
    roots = load_readings(args.summaries)

    cfg = adapter_config(args.charm_adapter)
    if not cfg.get("summaries"):
        raise SystemExit("STAR needs a CHARM adapter trained with preference readings")
    profiles = {}
    if cfg.get("profiles"):
        with open(args.profiles) as f:
            profiles = json.load(f)
    ce = CrossEncoderCHARM(args.base_model, device, max_length=cfg.get("max_length", 256),
                           dtype=dtype, heads=cfg.get("heads", 1), profiles=profiles or None)
    ce.load_adapter(args.charm_adapter)
    ce.train(False)
    local = os.path.isdir(args.lm)
    tok = AutoTokenizer.from_pretrained(args.lm, local_files_only=local)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(args.lm, dtype=dtype, local_files_only=local).to(device).eval()

    first = {}
    for i, r in enumerate(rows):
        first.setdefault(r["input"], i)
    dialogues = list(first)
    if args.limit:
        dialogues = dialogues[:args.limit]
    m = min(args.top_m, raw["top_idx"].size(1))
    cands = {d: [catalog[c] for c in raw["top_idx"][first[d], :m].tolist()] for d in dialogues}

    def propose(pairs):
        """One sampled reading per (dialogue, parent reading) pair."""
        if not pairs:
            return []
        prompts = [tok.apply_chat_template(reading_messages(d, parent), add_generation_prompt=True,
                                           tokenize=False) for d, parent in pairs]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                              temperature=args.temperature, top_p=0.95, pad_token_id=pad_id)
        return [clean_reading(t) for t in
                tok.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)]

    def charm(pairs):
        """CHARM scores of each (dialogue, reading) pair's shortlist, ``[len, M]``."""
        if not pairs:
            return torch.empty(0, m)
        return ce.score_groups([d for d, _ in pairs], [cands[d] for d, _ in pairs],
                               args.dialogues_per_batch, summaries=[r for _, r in pairs]).cpu()

    star = torch.zeros(len(dialogues), m)
    root_only = torch.zeros(len(dialogues), m)
    stats = {"nodes": 0, "pruned": 0, "backtracked": 0, "depth_sum": 0}
    examples, start = [], time.time()
    for s in range(0, len(dialogues), args.dialogues_per_batch):
        chunk = dialogues[s:s + args.dialogues_per_batch]
        root_readings = [roots.get(dialogue_key(d)) or "" for d in chunk]
        root_scores = charm(list(zip(chunk, root_readings)))
        frontiers = [[Node(rd, sc, node_value(sc))] for rd, sc in zip(root_readings, root_scores)]
        for depth in range(1, args.depth + 1):
            width = args.beam if depth == 1 else args.branch
            asks = [(j, node) for j, fr in enumerate(frontiers) for node in fr for _ in range(width)]
            texts = propose([(chunk[j], node.reading or None) for j, node in asks])
            scored = charm([(chunk[j], t) for (j, _), t in zip(asks, texts)])
            children = {}
            for (j, parent), text, sc in zip(asks, texts, scored):
                if text and text != parent.reading:
                    children.setdefault((j, id(parent)), []).append(
                        Node(text, sc, node_value(sc), depth, parent))
            for j, fr in enumerate(frontiers):
                kids = [children.get((j, id(p)), []) for p in fr]
                stats["nodes"] += sum(len(k) for k in kids)
                new = next_frontier(fr, kids, args.beam, args.backtrack)
                stats["pruned"] += sum(len(k) - len(survivors(k, p, args.backtrack))
                                       for p, k in zip(fr, kids))
                stats["backtracked"] += sum(1 for n in new if n.depth < depth)
                frontiers[j] = new
        for j, fr in enumerate(frontiers):
            star[s + j] = aggregate(fr, args.tau)
            root_only[s + j] = root_scores[j]
            stats["depth_sum"] += max(n.depth for n in fr)
            if len(examples) < 5 and j == 0:
                examples.append({"root": root_readings[j],
                                 "leaves": [(n.reading, round(n.value, 3)) for n in fr]})
        done = s + len(chunk)
        if (s // args.dialogues_per_batch) % 10 == 0:
            rate = (time.time() - start) / done
            print(f"  {done}/{len(dialogues)} dialogues (~{rate * (len(dialogues) - done) / 60:.0f} min left)",
                  flush=True)

    slot = {d: n for n, d in enumerate(dialogues)}
    k = raw["top_idx"].size(1)

    def widen(per_dialogue):
        out = torch.zeros(len(rows), k)
        covered = torch.zeros(len(rows), dtype=torch.bool)
        for i, r in enumerate(rows):
            if r["input"] in slot:
                row = per_dialogue[slot[r["input"]]]
                out[i, :m] = row
                out[i, m:] = row.mean()
                covered[i] = True
        return out, covered

    scores, covered = widen(star)
    roots_wide, _ = widen(root_only)
    hit = raw["top_idx"][covered, :m] == targets[covered, None]
    in_m = hit.any(1)

    def top1(x):
        return 100 * float(((x[covered][:, :m].argmax(1) == hit.float().argmax(1)) & in_m).float().mean())

    n_d = max(len(dialogues), 1)
    print(f"STAR in {(time.time() - start) / 60:.1f} min over {len(dialogues)} dialogues: "
          f"{stats['nodes'] / n_d:.1f} nodes/dialogue, {stats['pruned'] / n_d:.1f} pruned, "
          f"{stats['backtracked'] / n_d:.2f} backtracks, mean final depth {stats['depth_sum'] / n_d:.2f}")
    print(f"top-1 within top-{m} (target there {100 * float(in_m.float().mean()):.1f}%): "
          f"search {top1(scores):.2f}%  vs root reading only {top1(roots_wide):.2f}%", flush=True)
    for e in examples[:3]:
        print(f"  root: {e['root']!r}\n    leaves: {e['leaves']}")
    torch.save({"scores": scores, "root_scores": roots_wide, "targets": targets, "covered": covered,
                "top_m": m, "stats": stats, "examples": examples,
                "args": vars(args)}, args.out + ".partial")
    os.replace(args.out + ".partial", args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
