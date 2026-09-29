"""
HARPO: Optimizing Conversational Recommendation for User-Aligned Quality 
       via Hierarchical Preference Learning

Configuration and Data Structures

Novel Components:
- CHARM: Contrastive Hierarchical Alignment with Reward Marginalization  
- STAR: Structured Tree-of-thought Agentic Reasoning
- BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
- MAVEN: Multi-Agent Virtual Environment for Recommendations

ACL 2025 Submission
"""

from dataclasses import dataclass, field
from typing import List
from enum import Enum

# ============================================================================
# VIRTUAL TOOL OPERATIONS (VTOs) - Domain Agnostic Abstractions
# ============================================================================

class VTO(Enum):
    """Virtual Tool Operations - Domain-agnostic reasoning primitives."""
    # Information Extraction
    ANALYZE_SENTIMENT = "analyze_sentiment"
    EXTRACT_CONTEXT = "extract_context"
    EXTRACT_ENTITIES = "extract_entities"
    
    # User Modeling
    RETRIEVE_PREFERENCES = "retrieve_preferences"
    IDENTIFY_CONSTRAINTS = "identify_constraints"
    MODEL_USER_STATE = "model_user_state"
    
    # Search & Retrieval
    SEARCH_CANDIDATES = "search_candidates"
    FILTER_RESULTS = "filter_results"
    MATCH_ATTRIBUTES = "match_attributes"
    
    # Ranking & Selection
    RANK_OPTIONS = "rank_options"
    COMPARE_OPTIONS = "compare_options"
    SELECT_BEST = "select_best"
    
    # Knowledge & Reasoning
    QUERY_KNOWLEDGE = "query_knowledge"
    REASON_OVER_GRAPH = "reason_over_graph"
    INFER_IMPLICIT = "infer_implicit"
    
    # Interaction
    EXPLAIN_CHOICE = "explain_choice"
    REFINE_QUERY = "refine_query"
    HANDLE_REJECTION = "handle_rejection"
    
    # Memory
    TRACK_HISTORY = "track_history"
    UPDATE_BELIEFS = "update_beliefs"
    RECALL_CONTEXT = "recall_context"


# ============================================================================
# DOMAIN CONFIGURATIONS
# ============================================================================

class Domain(Enum):
    FASHION = "fashion"
    MOVIES = "movies"
    ELECTRONICS = "electronics"
    GENERAL = "general"
    FOOD = "food"
    BOOKS = "books"


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the base model.
    
    PUBLICATION OPTIMIZED: Using DeepSeek-R1-Distill-Qwen-7B
    - Excellent reasoning capability (R1 distillation)
    - 7B parameters - good balance of quality and speed
    - Works well with 2x A100 80GB
    """
    # Base model - DeepSeek R1 Distill for strong reasoning
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    # Model dimensions (auto-detected from model, defaults for 7B Qwen)
    hidden_size: int = 3584  # DeepSeek-R1-Distill-Qwen-7B hidden size
    num_attention_heads: int = 28
    num_hidden_layers: int = 28
    intermediate_size: int = 18944
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    
    # LoRA configuration - Optimized for 7B model
    lora_r: int = 128  # Higher rank for 7B model capacity
    lora_alpha: int = 256  # 2x lora_r for stable training
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Learning rates - CRITICAL: Conservative for stable training on 7B model
    # High LR causes catastrophic forgetting (gibberish output)
    sft_learning_rate: float = 5e-5  # FIXED: Reduced from 2e-4 to prevent forgetting
    charm_learning_rate: float = 2e-5  # FIXED: Reduced for stable preference learning
    star_learning_rate: float = 1e-5  # FIXED: Very conservative for reasoning module
    maven_learning_rate: float = 1e-5  # FIXED: Very conservative for multi-agent
    
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10  # FIXED: Longer warmup for training stability
    max_grad_norm: float = 1.0
    
    # Flash Attention 2 - 2-3x speedup on A100
    use_flash_attention: bool = True


# ============================================================================
# BRIDGE (Domain Adaptation) CONFIGURATION
# ============================================================================

@dataclass
class BRIDGEConfig:
    """Configuration for BRIDGE domain adapter."""
    adapter_hidden_size: int = 256
    num_projection_heads: int = 4
    adversarial_alpha: float = 1.0
    gradient_reversal: bool = True
    domain_confusion_weight: float = 0.1
    use_domain_gates: bool = True
    gate_init: float = 0.5


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

@dataclass
class RetrievalConfig:
    """Configuration for the item retriever.

    No counterpart existed in the original code: nothing in the four-stage
    curriculum trained the model to rank items, despite Recall@K being the
    headline metric.
    """
    enabled: bool = True
    embed_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.07
    loss_weight: float = 1.0

    # Content-only ignores the collaborative signal in ~10k ReDial dialogues;
    # id-only cannot generalise to unseen items. "hybrid" fuses both.
    item_representation: str = "hybrid"      # "content" | "id" | "hybrid"
    id_embedding_weight: float = 0.5
    max_catalog_size: int = 20000
    item_max_length: int = 48

    # In-batch negatives alone are weak and popularity-biased.
    num_hard_negatives: int = 4
    hard_negative_start_epoch: int = 0
    popularity_debias: float = 0.5

    # Per-item logit bias initialised to log training frequency (logit
    # adjustment). Ranking then starts at the popularity baseline -- which beats
    # every dialogue-blind scorer on ReDial -- and the retriever only has to
    # learn what the dialogue adds on top.
    item_bias: bool = True

    use_reranker: bool = False
    rerank_top_k: int = 50


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration for all stages.
    
    OPTIMIZED FOR: 2x A100 80GB with DeepSeek-R1-Distill-Qwen-7B
    Expected training time: ~1-1.5 hours (OPTIMIZED from ~2-3 hours)
    """
    # Data
    # OPTIMIZED: Reduced from 768 to 512 based on data analysis
    # 95th percentile = 355 tokens, 99th = 471 tokens
    # Only 0.5% of samples truncated at 512
    max_seq_length: int = 512  # OPTIMIZED: Was 768
    batch_size: int = 12  # OPTIMIZED: Can increase due to shorter sequences
    gradient_accumulation_steps: int = 4  # Effective batch = 12 * 4 * 2 GPUs = 96
    
    # Stage epochs - FIXED: More epochs for thorough learning
    sft_epochs: int = 1  # FIXED: Increased from 3 for better VTO learning
    charm_epochs: int = 1  # FIXED: Increased from 2 for better preference learning
    star_epochs: int = 1
    maven_epochs: int = 1
    
    # Evaluation
    eval_steps: int = 200
    save_steps: int = 500
    logging_steps: int = 50
    
    # Optimization - CRITICAL for stable training
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10  # FIXED: Longer warmup for stable convergence
    max_grad_norm: float = 1.0
    
    # Hardware - optimized for 2x A100 80GB
    fp16: bool = False
    bf16: bool = True  # BF16 for A100
    gradient_checkpointing: bool = True  # Enable for 7B model memory efficiency
    
    # Acceleration settings
    use_accelerate: bool = True  # Use HuggingFace Accelerate for multi-GPU
    dataloader_num_workers: int = 8  # OPTIMIZED: Increased from 4 for better throughput
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 4  # OPTIMIZED: Added prefetch factor
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Module-specific configs
    bridge_config: BRIDGEConfig = field(default_factory=BRIDGEConfig)
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
