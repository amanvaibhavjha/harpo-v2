#!/usr/bin/env python3
"""
Satisfaction and engagement labels for CHARM (see harpo/charm_labels.py).

Reads the raw ReDial release (train_data.jsonl, test_data.jsonl; downloaded by
scripts/convert_redial.py into redial_raw/) and writes
``{conversation_id: {title (lower case): {"liked": 0/1/null, "engaged": 0/1/null}}}``.

    python scripts/redial_labels.py --raw-dir redial_raw --out labels.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=os.path.join(os.path.dirname(__file__), "..", "redial_raw"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from harpo.charm_labels import conversation_labels

    labels, counts = {}, {"liked": [0, 0, 0], "engaged": [0, 0, 0]}   # 0, 1, unknown
    for name in ("train_data.jsonl", "test_data.jsonl"):
        path = os.path.join(args.raw_dir, name)
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run scripts/convert_redial.py once to download ReDial")
        with open(path) as f:
            for line in f:
                conv = json.loads(line)
                per_movie = conversation_labels(conv)
                labels[str(conv["conversationId"])] = {t.lower(): v for t, v in per_movie.items()}
                for v in per_movie.values():
                    for key in counts:
                        counts[key][2 if v[key] is None else v[key]] += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(labels, f)
    for key, (neg, pos, unk) in counts.items():
        n = neg + pos
        print(f"{key:<9} labelled {n} ({100 * pos / max(n, 1):.1f}% positive), unknown {unk}")
    print(f"wrote {args.out}: {len(labels)} conversations")


if __name__ == "__main__":
    main()
