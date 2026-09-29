import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_sample_group_puts_positive_first_and_excludes_turn_positives():
    import random
    from harpo.charm_ce import sample_group
    rng = random.Random(0)
    g = sample_group(5, [5, 1, 2, 3, 4, 9], exclude=[9], group=4, num_items=100, rng=rng)
    assert g[0] == 5 and len(g) == 4 and len(set(g)) == 4
    assert 9 not in g and all(c in {1, 2, 3, 4} for c in g[1:])


def test_sample_group_tops_up_with_random_items_when_shortlist_is_short():
    import random
    from harpo.charm_ce import sample_group
    g = sample_group(0, [0, 1], exclude=[], group=6, num_items=50, rng=random.Random(1))
    assert g[0] == 0 and g[1] == 1 and len(set(g)) == 6


def test_fused_ranks_weight_zero_reproduces_retriever():
    from harpo.charm_ce import fused_ranks
    retr = torch.tensor([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    other = torch.tensor([[0.0, 0.0, 9.0], [9.0, 0.0, 0.0]])
    tpos = torch.tensor([2, 0])
    rank = torch.tensor([99.0, 99.0])
    assert fused_ranks(retr, other, 0.0, tpos, rank).tolist() == [3.0, 3.0]
    assert fused_ranks(retr, other, 1.0, tpos, rank).tolist() == [1.0, 1.0]
    # target outside the shortlist keeps its retriever rank
    assert fused_ranks(retr, other, 0.5, torch.tensor([-1, 0]), torch.tensor([77.0, 5.0]))[0] == 77.0
