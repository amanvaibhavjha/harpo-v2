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
pip install -r requirements.txt
bash reproduce.sh
```

The data ships in `data/` (see Data below) and is checksummed on every run.
See the header of `reproduce.sh` for GPUs, models and resuming (about 27 h on
one A100 80 GB).

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
