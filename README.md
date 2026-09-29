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

The converted ReDial data ships in `data/redial_data.tar.gz`; the original
ReDial release (the seeker answers behind CHARM's satisfaction and engagement
labels) is downloaded on the first run. See the header of `reproduce.sh` for
GPUs, models and resuming (about 27 h on one A100 80 GB).

## Layout

- `harpo/` modules
- `scripts/` pipeline stages, in the order `reproduce.sh` runs them
- `tests/` unit tests (`python -m pytest tests`)

## Data

ReDial (Li et al., 2018), released under CC BY 4.0. `data/redial_data.tar.gz` is
derived from it by `scripts/convert_redial.py`.

## License

Code: MIT (`LICENSE`). Data: ReDial, CC BY 4.0.

## Citation

```bibtex
@inproceedings{raj2026harpo,
  title     = {{HARPO}: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation},
  author    = {Raj, Subham and Jha, Aman Vaibhav and Anand, Mayank and Saha, Sriparna},
  booktitle = {Proceedings of ACL 2026},
  year      = {2026},
  eprint    = {2604.10048},
  archivePrefix = {arXiv}
}

@inproceedings{li2018redial,
  title     = {Towards Deep Conversational Recommendations},
  author    = {Li, Raymond and Kahou, Samira Ebrahimi and Schulz, Hannes and Michalski, Vincent and Charlin, Laurent and Pal, Chris},
  booktitle = {Advances in Neural Information Processing Systems 31},
  year      = {2018}
}
```
