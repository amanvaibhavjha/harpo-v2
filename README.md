# HARPO

Conversational recommendation with four modules on a two-tower retriever:
CHARM (cross-encoder re-ranker: relevance, satisfaction, engagement, diversity),
STAR (value-guided tree search over readings of the seeker's wishes),
BRIDGE (LLM-written item profiles) and MAVEN (consensus of the agents).

ReDial, standard test protocol (4,975 cases), full 6,630-movie catalogue:

| R@1 | R@10 | R@50 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| 8.80 | 30.45 | 50.09 | 18.37 | 14.65 |

`results/` holds the test metrics (`maven.json`), the ablation without CHARM's
stage-1 agent (`maven_pure.json`), each agent alone (`agents.json`) and CHARM's
validation history (`charm_stage2.json`).

## Reproduce

```bash
conda create -n harpo python=3.10 -y && conda activate harpo
pip install -r requirements.txt
source harpo.env   # 1 GPU by default; edit it first if you have 2 (see below)
bash reproduce.sh
```

The data ships in `data/` (see Data below) and is checksummed on every run.
See the header of `reproduce.sh` for models and resuming (about 27 h on one
A100 80 GB, 16 h on two).

Paths, models and GPUs are configured in [`harpo.env`](harpo.env). It
defaults to a single GPU (`GPU0=GPU1=0`, everything sequential). If you have
two, set `GPU1=1` in `harpo.env` (or `GPU1=1 bash reproduce.sh` on the fly)
to run two stages in parallel. **If you skip `source harpo.env` and just run
`bash reproduce.sh` directly, the script uses its own built-in default of
`GPU1=1`** — i.e. it assumes 2 GPUs and will fail on a single-GPU machine
trying to use a GPU 1 that doesn't exist.

Prefer to run each stage yourself instead of the shell script (e.g. in a
`harpo` conda env)? See [MANUAL_STEPS.md](MANUAL_STEPS.md) for the same
pipeline unrolled into individual commands.

## Layout

- `harpo/` modules
- `scripts/` pipeline stages, in the order `reproduce.sh` runs them
- `tests/` unit tests (`python -m pytest tests`)

## Data

ReDial (Li et al., 2018), released under CC BY 4.0.

- `data/redial_raw.tar.gz`: the original ReDial release (`train_data.jsonl`,
  `test_data.jsonl`).
- `data/redial_data.tar.gz`: its conversion by `scripts/convert_redial.py`, the
  data behind the reported numbers. It is the rule-based conversion
  (`used_llm: false` in `stats.json`): dialogues, roles and target movies come
  straight from ReDial; the reasoning annotations in the replies (the operation
  lists inside `<|think|>`) come from keyword rules. A conversion with
  GPT-4o-mini-written annotations (same dialogues and targets) exists but was
  not used for these results.


## Citation

```bibtex
@inproceedings{raj2026harpo,
  title={HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation},
  author={Raj, Subham and Jha, Aman Vaibhav and Anand, Mayank and Saha, Sriparna},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={35580--35599},
  year={2026}
}
```
