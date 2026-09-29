"""
STAR: value-guided tree search over readings of what the seeker wants, with
CHARM as the value network.

A *reading* is one sentence describing the seeker's wishes ("wants a light
90s comedy; liked Groundhog Day"). CHARM scores the shortlist with a reading
appended to the dialogue, so every reading induces a ranking. The search:

  1. root: the greedy reading (the one CHARM was trained with);
  2. expand: the LLM proposes alternative or more specific readings of the
     same conversation (sampled);
  3. evaluate: CHARM re-scores the shortlist under each reading; the node's
     value is how decisively CHARM separates a winner,
     ``1 - H(softmax(scores)) / log M``;
  4. backtrack: a child whose value falls more than ``backtrack`` (0.3) below
     its parent's is pruned; a parent with no surviving child stays in play;
  5. beam: the ``beam`` (3) best nodes go one level deeper, to ``depth`` (3).

STAR's scores are the leaves' z-scored CHARM scores, averaged with weights
``softmax(value / tau)``. The reasoning therefore changes the ranking directly,
and ambiguous requests are covered by several readings instead of one.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch


@dataclass
class Node:
    reading: str
    scores: torch.Tensor                 # [M] CHARM scores of the shortlist under this reading
    value: float
    depth: int = 0
    parent: Optional["Node"] = field(default=None, repr=False)


def node_value(scores: torch.Tensor) -> float:
    """``1 - H(softmax(scores)) / log M``: 0 for a flat ranking, 1 for a certain one."""
    p = torch.softmax(scores.float(), dim=-1)
    h = float(-(p * p.clamp(min=1e-12).log()).sum())
    return 1.0 - h / math.log(max(scores.numel(), 2))


def survivors(children: Sequence[Node], parent: Node, backtrack: float) -> List[Node]:
    """Children whose value is at least ``(1 - backtrack)`` of the parent's."""
    return [c for c in children if c.value >= (1.0 - backtrack) * parent.value]


def next_frontier(frontier: Sequence[Node], children_of: Sequence[Sequence[Node]],
                  beam: int, backtrack: float) -> List[Node]:
    """The ``beam`` best of the surviving children, keeping any parent whose
    children were all pruned (backtracking to it)."""
    pool: List[Node] = []
    for parent, children in zip(frontier, children_of):
        kept = survivors(children, parent, backtrack)
        pool.extend(kept if kept else [parent])
    unique, seen = [], set()
    for n in sorted(pool, key=lambda n: -n.value):
        if n.reading not in seen:
            seen.add(n.reading)
            unique.append(n)
    return unique[:beam]


def aggregate(leaves: Sequence[Node], tau: float = 0.05) -> torch.Tensor:
    """``[M]``: value-weighted mean of the leaves' z-scored CHARM scores."""
    z = torch.stack([(n.scores - n.scores.mean()) / n.scores.std().clamp(min=1e-6)
                     for n in leaves])
    w = torch.softmax(torch.tensor([n.value for n in leaves]) / tau, dim=0)
    return (w[:, None] * z).sum(0)
