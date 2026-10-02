import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_dialogue_diversity_target_none_without_history():
    from harpo.diversity import dialogue_diversity_target, profile_vectors
    vecs = profile_vectors(["A"], {"A": "action thriller"})
    assert dialogue_diversity_target(0, [], vecs) is None


def test_dialogue_diversity_target_ranks_similar_below_dissimilar():
    from harpo.diversity import dialogue_diversity_target, profile_vectors
    titles = ["Action A", "Action B", "Romance C"]
    profiles = {
        "Action A": "action thriller explosions car chase",
        "Action B": "action thriller explosions gun fight",
        "Romance C": "romance drama wedding love story",
    }
    vecs = profile_vectors(titles, profiles)
    similar = dialogue_diversity_target(1, [0], vecs)    # two action movies
    dissimilar = dialogue_diversity_target(2, [0], vecs)  # action vs. romance
    assert 0.0 <= similar <= dissimilar <= 1.0
    assert dissimilar - similar > 0.5  # clearly separated, not just noise
