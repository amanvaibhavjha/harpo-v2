# HARPO

Conversational recommendation with four modules on a two-tower retriever:
CHARM (cross-encoder re-ranker: relevance, satisfaction, engagement, diversity),
STAR (value-guided tree search over readings of the seeker's wishes),
BRIDGE (LLM-written item profiles) and MAVEN (consensus of the agents).

ReDial, standard test protocol (4,975 cases), full 6,630-movie catalogue:

| R@1 | R@10 | R@50 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| 8.80 | 30.45 | 50.09 | 18.37 | 14.65 |

## Reproduce

```bash
pip install -r requirements.txt
RAW=/path/to/redial_raw bash reproduce.sh
```

`RAW` holds the original ReDial release (`train_data.jsonl`, `test_data.jsonl`).
The converted data ships in `data/redial_data.tar.gz`. See the header of
`reproduce.sh` for GPUs, models and resuming.

## Layout

- `harpo/` modules
- `scripts/` pipeline stages, in the order `reproduce.sh` runs them
- `tests/` unit tests (`python -m pytest tests`)
