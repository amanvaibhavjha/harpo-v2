"""
CHARM's fourth dimension, diversity.

Two distinct uses, both driven by the same TF-IDF similarity of BRIDGE
profiles (genre, era and tone words):

1. At inference, over the whole fused list (a property of the list, not of
   one candidate): the top is re-ranked by maximal marginal relevance (MMR),
   each next pick trading its (normalised) score against its similarity to
   the movies already placed above it. How strongly depends on the dialogue
   -- a vague request (a flat fused distribution) spreads the list over
   different kinds of movie, a precise one keeps the ranking:

       lambda(dialogue) = lambda0 * H(softmax(fused)) / log K

   with lambda0 chosen on validation (0, no diversity, is always a candidate).

2. At CHARM training, a per-candidate *proxy* target (`dialogue_diversity_target`)
   for its diversity head: how dissimilar the one movie actually recommended
   was from what the dialogue had already discussed. This is necessarily an
   approximation -- CHARM's other heads (relevance, satisfaction, engagement)
   score one candidate against the dialogue, which has a natural per-candidate
   meaning; diversity does not, so training it needs some per-candidate stand-in
   for the list-level property the head is meant to capture.
"""

import math
import re
from collections import Counter
from typing import Dict, Optional, Sequence

import torch

_WORD = re.compile(r"[a-z]+")
_STOP = set("a an the and or of to in on for with about from by as is are was were be its it "
            "his her their this that at into over who whom which film movie story tone".split())


def profile_vectors(titles: Sequence[str], profiles: Dict[str, str], min_df: int = 2) -> torch.Tensor:
    """``[N, V]`` L2-normalised TF-IDF vectors of each title's profile (zeros if none)."""
    docs = [[w for w in _WORD.findall((profiles.get(t) or "").lower()) if w not in _STOP]
            for t in titles]
    df = Counter(w for d in docs for w in set(d))
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= min_df))}
    n = len(docs)
    vecs = torch.zeros(n, max(len(vocab), 1))
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            j = vocab.get(w)
            if j is not None:
                vecs[i, j] = c * math.log(n / df[w])
    return torch.nn.functional.normalize(vecs, dim=-1)


def dialogue_diversity_target(candidate_idx: int, history_idx: Sequence[int],
                              vecs: torch.Tensor) -> Optional[float]:
    """CHARM diversity-head training target for one (dialogue, candidate) pair.

    1 minus the mean cosine similarity between the candidate's BRIDGE profile
    and the profiles of movies already mentioned earlier in the *same*
    dialogue: 0 if it reads like what was already discussed, 1 if unrelated.
    None if the dialogue has not mentioned any movie yet -- nothing to be
    diverse from, so the example carries no diversity signal and should be
    dropped from the loss, the same way a missing satisfaction/engagement
    label is.
    """
    if not history_idx:
        return None
    sims = vecs[list(history_idx)] @ vecs[candidate_idx]
    return float((1.0 - sims.mean()).clamp(0.0, 1.0))


def uncertainty(scores: torch.Tensor) -> torch.Tensor:
    """``[N]`` normalised entropy of ``softmax(scores)`` over each list (0 sharp, 1 flat)."""
    p = torch.softmax(scores.float(), dim=-1)
    h = -(p * p.clamp(min=1e-12).log()).sum(-1)
    return h / math.log(scores.size(-1))


def mmr_rerank(scores: torch.Tensor, items: torch.Tensor, vecs: torch.Tensor,
               lam: torch.Tensor, top: int = 30, chunk: int = 512) -> torch.Tensor:
    """Scores whose order is the MMR re-ranking of each row's top ``top``.

    ``scores``/``items`` are ``[N, K]`` (catalogue indices), ``lam`` is ``[N]``.
    Rows with ``lam == 0`` keep their order; items below the top keep their
    scores (and stay below every re-ranked item).
    """
    if scores.size(0) > chunk:
        return torch.cat([mmr_rerank(scores[i:i + chunk], items[i:i + chunk], vecs,
                                     lam[i:i + chunk], top, chunk)
                          for i in range(0, scores.size(0), chunk)])
    n, k = scores.shape
    top = min(top, k)
    order = scores.argsort(dim=1, descending=True)[:, :top]            # [N, L]
    s_top = scores.gather(1, order).float()
    lo, hi = s_top.min(1, keepdim=True).values, s_top.max(1, keepdim=True).values
    rel = (s_top - lo) / (hi - lo).clamp(min=1e-6)                      # [0, 1] per row
    v = vecs[items.gather(1, order)]                                     # [N, L, V]
    sim = torch.bmm(v, v.transpose(1, 2))                                # [N, L, L]
    max_sim = torch.zeros(n, top, device=scores.device)
    taken = torch.zeros(n, top, dtype=torch.bool, device=scores.device)
    rank_of = torch.zeros(n, top, device=scores.device)
    rows = torch.arange(n, device=scores.device)
    for step in range(top):
        mmr = rel - lam[:, None].float() * max_sim
        mmr[taken] = -float("inf")
        pick = mmr.argmax(1)
        taken[rows, pick] = True
        rank_of[rows, pick] = step
        max_sim = torch.maximum(max_sim, sim[rows, pick])
    out = scores.clone().float()
    base = scores.max(1, keepdim=True).values.float() + 1.0              # above everything else
    out.scatter_(1, order, base + (top - rank_of))
    return out


def intra_list_diversity(items: torch.Tensor, scores: torch.Tensor, vecs: torch.Tensor,
                         k: int = 10) -> float:
    """Mean pairwise (1 - cosine) among each row's top-``k`` items."""
    top = items.gather(1, scores.argsort(dim=1, descending=True)[:, :k])
    off = ~torch.eye(k, dtype=torch.bool, device=scores.device)
    total = 0.0
    for i in range(0, top.size(0), 512):
        v = vecs[top[i:i + 512]]
        total += float((1 - torch.bmm(v, v.transpose(1, 2))[:, off]).sum())
    return total / (top.size(0) * int(off.sum()))
