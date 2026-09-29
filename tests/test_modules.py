import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from harpo.charm_ce import sample_group
from harpo.charm_labels import conversation_labels, engaged_after
from harpo.diversity import intra_list_diversity, mmr_rerank, profile_vectors, uncertainty
from harpo.star_search import Node, aggregate, next_frontier, node_value, survivors


# ---- negatives

def test_strata_draw_from_rank_bands_and_fill_randomly():
    shortlist = list(range(100, 300))                     # 200 ranked candidates
    group = sample_group(7, shortlist, exclude=[100, 101], group=21, num_items=5000,
                         rng=random.Random(0), strata=[(50, 0.7), (200, 0.25)])
    negs = group[1:]
    assert group[0] == 7 and len(negs) == 20 and len(set(negs)) == 20
    top, mid = set(shortlist[:50]), set(shortlist[50:200])
    assert sum(n in top for n in negs) == 14                # 0.7 * 20
    assert sum(n in mid for n in negs) == 5                 # 0.25 * 20
    assert not {100, 101} & set(negs)                       # excluded (same conversation)
    assert sum(n not in top | mid for n in negs) == 1       # random catalogue item


def test_uniform_sampling_unchanged_without_strata():
    a = sample_group(1, list(range(10, 40)), [11], 8, 100, random.Random(3))
    b = sample_group(1, list(range(10, 40)), [11], 8, 100, random.Random(3))
    assert a == b and 11 not in a and all(10 <= n < 40 for n in a[1:])


# ---- satisfaction / engagement labels

def _msg(sender, text):
    return {"senderWorkerId": sender, "text": text}


def test_engagement_rules():
    msgs = [_msg(1, "You might like @5"), _msg(0, "What's it about?")]
    assert engaged_after(msgs, 0, "5", 0) == 1            # asks straight away
    msgs = [_msg(1, "Try @5"), _msg(0, "Not really interested, anything else")]
    assert engaged_after(msgs, 0, "5", 0) == 0
    msgs = [_msg(1, "Try @5"), _msg(0, "ok"), _msg(1, "Or @6"), _msg(0, "I'll watch @5")]
    assert engaged_after(msgs, 0, "5", 0) == 0            # window ends at the next suggestion
    assert engaged_after([_msg(1, "Try @5")], 0, "5", 0) is None


def test_conversation_labels_use_seeker_answers_first():
    conv = {"initiatorWorkerId": 0, "movieMentions": {"5": "Up (2009)", "6": "Heat (1995)"},
            "initiatorQuestions": {"5": {"liked": 0}, "6": {"liked": 2}},
            "respondentQuestions": {"5": {"liked": 1}, "6": {"liked": 1}},
            "messages": [_msg(0, "hi"), _msg(1, "How about @5?"), _msg(0, "sounds great"),
                         _msg(1, "Also @6"), _msg(0, "")]}
    labels = conversation_labels(conv)
    assert labels["Up (2009)"] == {"liked": 0, "engaged": 1}
    assert labels["Heat (1995)"] == {"liked": 1, "engaged": None}   # seeker said nothing after


# ---- diversity

def test_mmr_identity_at_zero_and_spreads_similar_items():
    profiles = {"a": "space opera adventure", "b": "space opera adventure sequel",
                "c": "quiet french romance", "d": "space war adventure"}
    titles = ["a", "b", "c", "d"]
    vecs = profile_vectors(titles, profiles, min_df=1)
    assert torch.allclose(vecs.norm(dim=1), torch.ones(4), atol=1e-5)
    scores = torch.tensor([[4.0, 3.9, 3.0, 3.8]])
    items = torch.tensor([[0, 1, 2, 3]])
    same = mmr_rerank(scores, items, vecs, torch.tensor([0.0]))
    assert same.argsort(descending=True).tolist() == scores.argsort(descending=True).tolist()
    spread = mmr_rerank(scores, items, vecs, torch.tensor([2.0]))
    # The near-duplicate sequel (b) drops from second to last; the romance moves up.
    assert spread.argsort(descending=True)[0].tolist() == [0, 3, 2, 1]
    assert intra_list_diversity(items, spread, vecs, k=2) > intra_list_diversity(items, scores, vecs, k=2)
    u = uncertainty(torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]))
    assert abs(float(u[0]) - 1) < 1e-5 and float(u[1]) < 0.01


# ---- STAR search

def _node(reading, scores, depth=0):
    s = torch.tensor(scores)
    return Node(reading, s, node_value(s), depth)


def test_value_bounds_and_backtracking():
    flat, sharp = torch.zeros(10), torch.tensor([20.0] + [0.0] * 9)
    assert abs(node_value(flat)) < 1e-6 and node_value(sharp) > 0.99
    parent = _node("p", [5.0, 0.0, 0.0, 0.0])
    weak, strong = _node("w", [0.1, 0.0, 0.0, 0.0], 1), _node("s", [6.0, 0.0, 0.0, 0.0], 1)
    assert survivors([weak, strong], parent, 0.3) == [strong]
    # A parent whose children are all pruned stays in play.
    other = _node("o", [4.0, 0.0, 0.0, 0.0])
    frontier = next_frontier([parent, other], [[weak], [strong]], beam=3, backtrack=0.3)
    assert [n.reading for n in frontier] == ["s", "p"]


def test_aggregate_prefers_the_confident_leaf():
    confident = _node("c", [9.0, 0.0, 0.0])
    unsure = _node("u", [0.0, 0.2, 0.1])
    agg = aggregate([confident, unsure], tau=0.05)
    assert int(agg.argmax()) == 0


# ---- helpers in the scripts

def test_cv_fold_is_stable_and_balanced():
    from run_experiment import cv_fold
    rows = [{"conversation_id": str(i), "input": ""} for i in range(2000)]
    folds = [cv_fold(r, 2) for r in rows]
    assert folds == [cv_fold(r, 2) for r in rows]
    assert 900 < sum(folds) < 1100


def test_binary_auc_and_conversation_movies():
    from train_charm_ce import binary_auc, conversation_movies
    assert binary_auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert binary_auc([0.1, 0.9], [1, 0]) == 0.0
    assert binary_auc([0.3], [1]) == 0.5
    index = {"x": 0, "y": 1, "z": 2}
    rows = [{"conversation_id": "7", "ground_truth_item": "X", "all_conversation_movies": "['Y']"},
            {"conversation_id": "7", "ground_truth_item": "Z", "all_conversation_movies": "[]"}]
    assert conversation_movies(rows, index) == {"7": {0, 1, 2}}


def test_denoise_keeps_movies_already_in_the_dialogue():
    from train_charm_ce import later_movies
    titles = ["Alien (1979)", "Heat (1995)", "Up (2009)"]
    conv = {"7": {0, 1, 2}}
    row = {"conversation_id": 7, "input": "User: I loved Alien (1979). Anything like it?"}
    # Alien is already discussed: it stays a candidate; Heat and Up come later.
    assert later_movies(conv, row, titles) == {1, 2}
    assert later_movies(conv, {"conversation_id": 8, "input": "hi"}, titles) == set()
