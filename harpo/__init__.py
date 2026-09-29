"""
HARPO: agentic conversational recommendation.

  retriever  two-tower retrieval over the full catalogue, with cold-start item
             vectors (model.py, training.py, retrieval.py, coldstart.py)
  CHARM      cross-encoder re-ranker: relevance, satisfaction and engagement heads
             mixed by a dialogue gate (charm_ce.py), plus list diversity (diversity.py)
  STAR       value-guided tree search over readings of the seeker's wishes
             (star.py, star_search.py)
  BRIDGE     LLM-written item profiles, read by CHARM and by the diversity step
  MAVEN      consensus of the agents, weights learned on validation (maven.py)
"""

__version__ = "2.0.0"
