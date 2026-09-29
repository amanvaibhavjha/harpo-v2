#!/usr/bin/env python3
"""
One-sentence preference readings for every dialogue (CHARM's reading input,
and the root of STAR's search; see harpo/star.py).

Greedy decoding from the dialogue alone (never the reply that names the
answer), keyed by dialogue text. Shard across GPUs with --shard i/n and pass the
shard files comma-separated to the consumers.

    python scripts/pref_summaries.py --model Qwen/Qwen2.5-7B-Instruct --shard 0/2 --out s0.json
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default=os.path.expanduser("~/Desktop/harpo-work/redial_data"))
    parser.add_argument("--shard", default="0/1", help="i/n: this process's share of dialogues")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0, help="dialogues (smoke tests)")
    parser.add_argument("--only", default=None,
                        help="JSON list of dialogue texts to cover (smoke tests)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from harpo.star import clean_reading, dialogue_key, reading_messages

    shard, n_shards = (int(x) for x in args.shard.split("/"))
    dialogues = set()
    for name in ("sft_data.json", "test_sft.json"):
        with open(os.path.join(args.data, name)) as f:
            for r in json.load(f):
                if r.get("ground_truth_item") not in (None, "", "None"):
                    dialogues.add(r["input"])
    if args.only:
        with open(args.only) as f:
            dialogues &= set(json.load(f))
    dialogues = sorted(d for d in dialogues if int(dialogue_key(d), 16) % n_shards == shard)
    if args.limit:
        dialogues = dialogues[:args.limit]
    dialogues.sort(key=len)                         # similar lengths batch together
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = os.path.isdir(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=local)
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        local_files_only=local).to(device).eval()
    print(f"device={device}  shard {shard}/{n_shards}: {len(dialogues)} dialogues", flush=True)

    out, start = {}, time.time()
    for s in range(0, len(dialogues), args.batch):
        chunk = dialogues[s:s + args.batch]
        prompts = [tok.apply_chat_template(reading_messages(d), add_generation_prompt=True,
                                           tokenize=False) for d in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = lm.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                              pad_token_id=pad_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].size(1):], skip_special_tokens=True)
        for d, t in zip(chunk, texts):
            out[dialogue_key(d)] = clean_reading(t)
        if (s // args.batch) % 20 == 0:
            done = s + len(chunk)
            rate = (time.time() - start) / done
            print(f"  {done}/{len(dialogues)}  (~{rate * (len(dialogues) - done) / 60:.0f} min left)"
                  f"  e.g. {out[dialogue_key(chunk[0])]!r}", flush=True)
    with open(args.out + ".partial", "w") as f:
        json.dump(out, f)
    os.replace(args.out + ".partial", args.out)
    print(f"wrote {args.out}: {len(out)} readings in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
