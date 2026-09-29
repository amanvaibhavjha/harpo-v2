"""
HARPO: Unified Model Architecture

Optimizing Conversational Recommendation for User-Aligned Quality 
via Hierarchical Preference Learning

Integrates all novel components:
- CHARM: Contrastive Hierarchical Alignment with Reward Marginalization
- STAR: Structured Tree-of-thought Agentic Reasoning
- BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
- MAVEN: Multi-Agent Virtual Environment for Recommendations

Primary Objective: USER-ALIGNED RECOMMENDATION QUALITY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union
from dataclasses import dataclass

from .pooling import masked_mean_pool
from .retrieval import CrossEncoderReranker, TwoTowerRetriever
from .config import VTO, Domain, ModelConfig, TrainingConfig, BRIDGEConfig


@dataclass
class ModelOutput:
    """Unified output format"""
    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    vto_logits: Optional[torch.Tensor] = None
    context_pooled: Optional[torch.Tensor] = None
    
# ============================================================================
# BRIDGE: Domain Adaptation Module
# ============================================================================

class GradientReversal(torch.autograd.Function):
    """Gradient reversal for adversarial domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


class BRIDGE(nn.Module):
    """
    BRIDGE: Bidirectional Reasoning-Informed Domain-Generalized Embeddings
    
    Creates domain-invariant representations while preserving task-specific info.
    Key innovations:
    - Adversarial training with domain-specific gates
    - Contrastive learning for cross-domain alignment (NEW)
    - Cross-attention fusion for recommendation relevance (NEW)
    """
    
    def __init__(self, hidden_size: int, config: BRIDGEConfig, 
                 num_domains: int = 6, num_vtos: int = 24):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_domains = num_domains
        
        # Multi-head projection
        self.head_dim = hidden_size // config.num_projection_heads
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, self.head_dim),
                nn.LayerNorm(self.head_dim),
                nn.GELU()
            ) for _ in range(config.num_projection_heads)
        ])
        self.head_combiner = nn.Linear(hidden_size, hidden_size)
        
        # Domain discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_domains)
        )
        
        # Task preserver (VTO prediction)
        self.task_preserver = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_vtos)
        )
        
        # Domain gates
        # One row per domain so a batch spanning several domains can be gated
        # per sample. The old ParameterDict keyed by Domain.value could only
        # apply a single domain to an entire batch.
        if config.use_domain_gates:
            self.domain_gates = nn.Parameter(
                torch.full((num_domains, hidden_size), float(config.gate_init)))
        
        # NEW: Contrastive projection for cross-domain alignment
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        self.contrastive_temp = nn.Parameter(torch.tensor(0.07))  # Learnable temperature
        
        # NEW: Cross-attention for recommendation-aware feature fusion
        self.cross_attn_query = nn.Linear(hidden_size, hidden_size // 4)
        self.cross_attn_key = nn.Linear(hidden_size, hidden_size // 4)
        self.cross_attn_value = nn.Linear(hidden_size, hidden_size)
        self.cross_attn_out = nn.Linear(hidden_size, hidden_size)
        
        self.output_norm = nn.LayerNorm(hidden_size)
    
    @staticmethod
    def _as_domain_ids(domain: Union[Domain, torch.Tensor],
                       batch_size: int, device) -> torch.Tensor:
        """Normalise a Domain enum or an id tensor to a ``[batch]`` LongTensor."""
        if isinstance(domain, torch.Tensor):
            ids = domain.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1 and batch_size > 1:
                ids = ids.expand(batch_size)
            return ids
        return torch.full((batch_size,), list(Domain).index(domain),
                          dtype=torch.long, device=device)

    def forward(self, hidden_states: torch.Tensor,
                domain: Union[Domain, torch.Tensor],
                alpha: float = 1.0, vto_labels: Optional[torch.Tensor] = None,
                enable_contrastive: bool = True,
                attention_mask: Optional[torch.Tensor] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for domain adaptation.
        
        Returns domain-invariant features and auxiliary losses.
        NEW: Includes contrastive projection and cross-attention fusion.
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # Masked, so padding cannot leak into the representation.
        pooled = masked_mean_pool(hidden_states, attention_mask)
        batch_size = pooled.size(0)
        domain_ids = self._as_domain_ids(domain, batch_size, device)
        
        # Multi-head projection
        head_outputs = [head(pooled) for head in self.projection_heads]
        combined = torch.cat(head_outputs, dim=-1)
        projected = self.head_combiner(combined)
        
        # Apply domain gate
        if self.config.use_domain_gates:
            gate = torch.sigmoid(self.domain_gates[domain_ids])
            domain_invariant = gate * projected + (1 - gate) * pooled
        else:
            domain_invariant = projected
        
        # NEW: Cross-attention fusion for recommendation-aware features
        # Query from domain-invariant, Key/Value from original
        q = self.cross_attn_query(domain_invariant)
        k = self.cross_attn_key(pooled)
        v = self.cross_attn_value(pooled)
        
        # Scaled dot-product attention
        attn_scale = (q.size(-1)) ** 0.5
        attn_weights = torch.softmax(torch.sum(q * k, dim=-1, keepdim=True) / attn_scale, dim=-1)
        attended = attn_weights * v
        fused = domain_invariant + 0.1 * self.cross_attn_out(attended)  # Residual connection
        
        domain_invariant = self.output_norm(fused)
        
        outputs = {"features": domain_invariant}
        
        # Domain confusion loss
        if self.config.gradient_reversal:
            reversed_features = gradient_reversal(domain_invariant, alpha)
        else:
            reversed_features = domain_invariant
        
        domain_logits = self.domain_discriminator(reversed_features)
        outputs["domain_loss"] = F.cross_entropy(domain_logits, domain_ids)
        
        # NEW: Contrastive projection for cross-domain alignment
        if enable_contrastive and batch_size > 1:
            contrastive_embeds = self.contrastive_proj(domain_invariant)
            contrastive_embeds = F.normalize(contrastive_embeds, dim=-1)
            outputs["contrastive_embeds"] = contrastive_embeds
            
            # InfoNCE-style loss (within-batch negatives)
            sim_matrix = torch.mm(contrastive_embeds, contrastive_embeds.t()) / self.contrastive_temp.clamp(min=0.01)
            labels = torch.arange(batch_size, device=device)
            outputs["contrastive_loss"] = F.cross_entropy(sim_matrix, labels)
        
        # Task preservation loss
        if vto_labels is not None:
            vto_logits = self.task_preserver(domain_invariant)
            outputs["task_loss"] = F.binary_cross_entropy_with_logits(vto_logits, vto_labels.float())
            outputs["vto_logits"] = vto_logits
        
        return outputs


# ============================================================================
# The retriever's backbone: base LLM + LoRA, BRIDGE, heads, two-tower retrieval
# ============================================================================

class HARPOMTv2(nn.Module):
    """
    HARPO-MT v2: retriever backbone.

    Integrates:
    - Base LLM + LoRA
    - BRIDGE: domain adaptation
    - Two-tower retriever, VTO head, recommendation head

    STAR, CHARM and MAVEN are separate pipeline stages
    (star_search.py, charm_ce.py, maven.py) that consume this backbone's
    pooled hidden states; they are not submodules of this class.
    """
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        
        self.base_model = None
        self.tokenizer = None
        self.device = None
        self._model_dtype = torch.float32  # CRITICAL FIX: Track model dtype
        
        hidden_size = model_config.hidden_size
        num_vtos = len(VTO)
        num_domains = len(Domain)
        
        self.bridge = BRIDGE(hidden_size, training_config.bridge_config, num_domains, num_vtos)

        # VTO prediction head
        self.vto_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_vtos)
        )
        
        # Recommendation head (predicts item relevance)
        self.recommendation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Item retriever: the missing contrastive objective.
        retrieval_config = getattr(training_config, "retrieval_config", None)
        self.retrieval_config = retrieval_config
        if retrieval_config is not None and retrieval_config.enabled:
            self.retriever = TwoTowerRetriever(
                hidden_size, embed_dim=retrieval_config.embed_dim,
                dropout=retrieval_config.dropout,
                init_temperature=retrieval_config.temperature,
            )
            if retrieval_config.item_representation in ("id", "hybrid"):
                self.item_id_embedding = nn.Embedding(
                    retrieval_config.max_catalog_size, retrieval_config.embed_dim
                )
                nn.init.normal_(self.item_id_embedding.weight, std=0.02)
            else:
                self.item_id_embedding = None
            # Zero until a catalogue is attached, when it is set to log popularity.
            if getattr(retrieval_config, "item_bias", False):
                self.item_bias = nn.Embedding(retrieval_config.max_catalog_size, 1)
                nn.init.zeros_(self.item_bias.weight)
            else:
                self.item_bias = None
            self.reranker = (CrossEncoderReranker(hidden_size, retrieval_config.dropout)
                             if retrieval_config.use_reranker else None)
        else:
            self.retriever = None
            self.item_id_embedding = None
            self.item_bias = None
            self.reranker = None

    def encode_item_embeddings(self, item_pooled: torch.Tensor,
                               item_indices: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """Item embeddings, fusing the learned id table with the text tower.

        Items outside the table (index < 0) fall back to content only rather
        than erroring.
        """
        if self.retriever is None:
            raise RuntimeError("retrieval is disabled in this configuration")

        embeds = self.retriever.encode_item(item_pooled)
        cfg = self.retrieval_config
        if self.item_id_embedding is None or item_indices is None:
            return embeds
        if cfg.item_representation == "content":
            return embeds

        known = (item_indices >= 0) & (item_indices < self.item_id_embedding.num_embeddings)
        if not bool(known.any()):
            return embeds

        safe = item_indices.clamp(min=0, max=self.item_id_embedding.num_embeddings - 1)
        id_embeds = F.normalize(self.item_id_embedding(safe), dim=-1)

        if cfg.item_representation == "id":
            fused = torch.where(known.unsqueeze(-1), id_embeds, embeds)
        else:
            w = cfg.id_embedding_weight
            fused = torch.where(known.unsqueeze(-1),
                                (1.0 - w) * embeds + w * id_embeds, embeds)
        return F.normalize(fused, dim=-1)

    @staticmethod
    def _per_example_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean NLL per example, ``[batch]``.

        ``outputs.loss`` is already reduced over the batch, so it cannot be a
        per-example signal; broadcasting it gave every row the same target.
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1), ignore_index=-100, reduction="none",
        ).view(shift_labels.shape)
        valid = (shift_labels != -100).to(nll.dtype)
        return (nll * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
    
    def load_base_model(self, device: str = None):
        """Load base LLM with LoRA, Flash Attention 2, and proper special token handling
        
        CRITICAL FIXES for resume/offline mode:
        - Detects if loading from local checkpoint
        - Uses local_files_only=True for offline environments
        - Handles tokenizer and embedding size correctly
        - Doesn't re-add special tokens when resuming
        
        OPTIMIZED FOR PUBLICATION:
        - Flash Attention 2 for 2-3x speedup
        - Proper gradient checkpointing
        - Efficient memory usage for 7B model on 2x A100
        """
        import os
        import json
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        model_path = self.model_config.model_name
        
        # ===== Detect if loading from local checkpoint =====
        is_local_path = os.path.isdir(model_path)
        is_lora_checkpoint = is_local_path and os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        print(f"Loading base model: {model_path}")
        print(f"Device: {device}")
        if is_local_path:
            print(f"✓ Loading from LOCAL path (offline mode enabled)")
        if is_lora_checkpoint:
            print(f"✓ Detected LoRA adapter checkpoint")
        
        # ===== Set offline mode for HuggingFace =====
        # Only when genuinely loading from disk: hard-coding these made the
        # documented "fresh start (requires internet)" branch impossible.
        if is_local_path:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # ===== Determine dtype and check for Flash Attention 2 =====
        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        
        # Check Flash Attention 2 availability
        use_flash_attn = False
        if device.startswith("cuda") and getattr(self.model_config, 'use_flash_attention', True):
            try:
                import flash_attn
                use_flash_attn = True
                print("=" * 50)
                print("✓ FLASH ATTENTION 2 ENABLED - 2-3x speedup!")
                print("=" * 50)
            except ImportError:
                print("=" * 50)
                print("⚠ Flash Attention 2 NOT installed")
                print("  Install with: pip install flash-attn --no-build-isolation")
                print("  Training will work but be SLOWER (no flash attention)")
                print("=" * 50)
        
        # ===== Load tokenizer =====
        # A local path is not necessarily a HARPO checkpoint: a plain model
        # directory (a hub snapshot copied to a cluster node) has no special
        # tokens. Assuming it did skipped them entirely, so the same data was
        # tokenized differently depending on where the weights came from. The
        # missing-token check below is a no-op for checkpoints that have them.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            local_files_only=is_local_path
        )
        # Dialogues overflow from the start: the latest turns carry the request,
        # and right-truncation also cut the reply off 15% of SFT examples.
        self.tokenizer.truncation_side = "left"

        # ===== Add ALL special tokens used in training data =====
        special_tokens_to_add = [
            "<|vto_start|>", "<|vto_end|>",
            "<|tool_start|>", "<|tool_end|>",
            "<|think|>", "<|/think|>",
            "<|thought|>", "<|/thought|>",
            "<|response|>", "<|/response|>",
            "<|domain:fashion|>", "<|domain:movies|>", 
            "<|domain:electronics|>", "<|domain:general|>",
            "<|domain:food|>", "<|domain:books|>",
            "<|agent:recommender|>", "<|agent:critic|>", 
            "<|agent:explainer|>", "<|agent:orchestrator|>",
        ]

        # Check which tokens are truly new
        new_tokens = []
        for token in special_tokens_to_add:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)

        if new_tokens:
            num_added = self.tokenizer.add_special_tokens({
                'additional_special_tokens': new_tokens
            })
            print(f"✓ Added {num_added} special tokens to tokenizer")

        # Add pad token if missing (for both cases)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # ===== Load model =====
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "local_files_only": is_local_path,
        }
        
        # Add Flash Attention 2 if available
        if use_flash_attn:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        print(f"Loading model (this may take a moment for 7B params)...")
        
        if is_lora_checkpoint:
            # ===== CRITICAL: Loading from LoRA checkpoint requires special handling =====
            # 1. First load the BASE model from cache
            # 2. Resize embeddings to match tokenizer
            # 3. Then load LoRA adapter
            
            # Get the base model name from adapter config
            adapter_config_path = os.path.join(model_path, "adapter_config.json")
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
            
            print(f"Loading base model from cache: {base_model_name}")
            
            # Load base model from cache
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **load_kwargs
            )
            
            # Move to device BEFORE resizing
            if device.startswith("cuda"):
                target_device = device
                self.base_model = self.base_model.to(target_device)
            elif device == "mps":
                target_device = device
                self.base_model = self.base_model.to(device)
            else:
                target_device = device
            self._target_device = target_device
            
            # Resize embeddings to match tokenizer BEFORE loading LoRA
            old_vocab_size = self.base_model.get_input_embeddings().weight.shape[0]
            new_vocab_size = len(self.tokenizer)
            
            if old_vocab_size != new_vocab_size:
                self.base_model.resize_token_embeddings(new_vocab_size)
                print(f"✓ Resized embeddings: {old_vocab_size} -> {new_vocab_size}")
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {model_path}")
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                model_path,
                is_trainable=True,
                local_files_only=True
            )
            print(f"✓ Loaded LoRA adapter successfully")
            
            # Don't apply LoRA again - already loaded
            self._lora_applied = True
            
        else:
            # Fresh start or full model checkpoint
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            # Move to device
            if device.startswith("cuda"):
                target_device = device
                self.base_model = self.base_model.to(target_device)
            elif device == "mps":
                target_device = device
                self.base_model = self.base_model.to(device)
            else:
                target_device = device
            self._target_device = target_device
            
            # ===== Resize embeddings for new tokens =====
            if new_tokens:
                old_embeddings_size = self.base_model.get_input_embeddings().weight.shape[0]
                self.base_model.resize_token_embeddings(len(self.tokenizer))
                new_embeddings_size = self.base_model.get_input_embeddings().weight.shape[0]
                print(f"✓ Resized embeddings: {old_embeddings_size} -> {new_embeddings_size}")

                # Initialise the rows of the tokens actually added, by id rather
                # than by the change in matrix size. The size delta is the wrong
                # signal: Qwen2.5 pads its vocabulary to 151936 while holding
                # 151665 real tokens, so adding 20 tokens *shrinks* the matrix to
                # 151685 and range(old, new) is empty -- the old loop printed
                # success while initialising nothing, and the new ids silently
                # inherited unused reserved rows.
                new_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
                new_ids = [i for i in new_ids
                           if isinstance(i, int) and 0 <= i < new_embeddings_size]

                with torch.no_grad():
                    in_w = self.base_model.get_input_embeddings().weight
                    keep = torch.ones(new_embeddings_size, dtype=torch.bool)
                    if new_ids:
                        keep[torch.tensor(new_ids, dtype=torch.long)] = False
                    mean_embedding = in_w[keep].mean(dim=0)
                    for i in new_ids:
                        in_w[i] = (mean_embedding
                                   + torch.randn_like(mean_embedding) * 0.02).to(in_w.dtype)

                    # The untied output head needs the same treatment, or the
                    # model can never assign probability to the new tokens.
                    out_emb = self.base_model.get_output_embeddings()
                    if out_emb is not None and out_emb.weight is not in_w:
                        out_w = out_emb.weight
                        out_mean = out_w[keep].mean(dim=0)
                        for i in new_ids:
                            out_w[i] = (out_mean
                                        + torch.randn_like(out_mean) * 0.02).to(out_w.dtype)

                print(f"✓ Initialized {len(new_ids)} new token embeddings")
            
            # Apply LoRA for fresh start
            self._lora_applied = False
        
        # Enable gradient checkpointing for memory efficiency
        # CRITICAL FIX: Use use_reentrant=False for DDP compatibility
        if self.training_config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("✓ Gradient checkpointing enabled (non-reentrant mode for DDP)")
        
        # Apply LoRA if not already done
        if not getattr(self, '_lora_applied', False):
            self._apply_lora()
        
        # ===== CRITICAL FIX: Ensure PEFT modules_to_save are on correct device =====
        target_device = getattr(self, '_target_device', device)
        if hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'model'):
            base_transformer = self.base_model.base_model.model
            if hasattr(base_transformer, 'embed_tokens'):
                base_transformer.embed_tokens = base_transformer.embed_tokens.to(target_device)
        
        if hasattr(self.base_model, 'lm_head'):
            self.base_model.lm_head = self.base_model.lm_head.to(target_device)
        elif hasattr(self.base_model, 'base_model') and hasattr(self.base_model.base_model, 'lm_head'):
            self.base_model.base_model.lm_head = self.base_model.base_model.lm_head.to(target_device)
        
        # Move entire model to ensure all submodules are on correct device
        self.base_model = self.base_model.to(target_device)

        # All components must be on the same device as the base model.
        target_device = getattr(self, '_target_device', "cuda:0" if device.startswith("cuda") else device)

        # Move components to device AND convert to same dtype as base model
        self.bridge = self.bridge.to(target_device, dtype=model_dtype)
        self.vto_head = self.vto_head.to(target_device, dtype=model_dtype)
        self.recommendation_head = self.recommendation_head.to(target_device, dtype=model_dtype)
        # Omitting these left the towers on CPU while the backbone sat on the
        # accelerator, failing on any non-CPU device -- CUDA included.
        # Retrieval heads stay float32 on a bf16 backbone: an AdamW step (~lr)
        # is below bf16 resolution for most of their weights and is rounded
        # away -- a popularity bias near 6 would never move at all.
        for _name in ("retriever", "item_id_embedding", "item_bias", "reranker"):
            _mod = getattr(self, _name, None)
            if _mod is not None:
                setattr(self, _name, _mod.to(target_device, dtype=torch.float32))
        
        # Store dtype for _reinit_components
        self._model_dtype = model_dtype
        
        # Update hidden size from loaded model
        actual_hidden = self.base_model.config.hidden_size
        if actual_hidden != self.model_config.hidden_size:
            print(f"Updating hidden size: {self.model_config.hidden_size} -> {actual_hidden}")
            self.model_config.hidden_size = actual_hidden
            self._reinit_components(actual_hidden)
        
        print(f"✓ Model loaded. Trainable params: {self.count_parameters():,}")
        return self
    
    def _apply_lora(self):
        """Apply LoRA adapters
        
        CRITICAL FIX: Added modules_to_save to train embedding layers.
        Without this, new special tokens (<|think|>, <|tool_start|>, etc.) 
        would remain frozen and produce gibberish!
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            lora_config = LoraConfig(
                r=self.model_config.lora_r,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.lora_target_modules,
                lora_dropout=self.model_config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                # CRITICAL: Train embeddings for new special tokens!
                # Without this, tokens like <|think|> remain random noise
                # ~40% of a 0.5B model becomes trainable (two 151k x hidden
                # matrices), which dominates wall-clock on memory-bound devices.
                modules_to_save=(None if getattr(self, "_skip_modules_to_save", False)
                                 else ["embed_tokens", "lm_head"])
            )
            
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()
            print("✓ LoRA applied with trainable embeddings for special tokens")
            
        except ImportError:
            print("PEFT not available, using full fine-tuning (last 2 layers)")
            for name, param in self.base_model.named_parameters():
                if "layers.23" not in name and "layers.22" not in name:
                    param.requires_grad = False
    
    def _reinit_components(self, hidden_size: int):
        """Reinitialize components with correct hidden size.
        
        CRITICAL FIX: Only reinitialize if dimensions ACTUALLY changed.
        Previously this was destroying learned weights on every call!
        """
        num_vtos = len(VTO)
        num_domains = len(Domain)
        
        target_device = getattr(self, '_target_device', "cuda:0" if self.device.startswith("cuda") else self.device)
        
        # Use stored dtype (bfloat16 for A100)
        dtype = getattr(self, '_model_dtype', torch.float32)
        
        # CRITICAL FIX: Check if dimensions actually changed before reinitializing
        # This prevents destroying learned weights when loading checkpoints
        bridge_hidden = getattr(self.bridge, 'hidden_size', None)
        if bridge_hidden is None or bridge_hidden != hidden_size:
            print(f"  Reinitializing BRIDGE: {bridge_hidden} -> {hidden_size}")
            self.bridge = BRIDGE(hidden_size, self.training_config.bridge_config, num_domains, num_vtos).to(target_device, dtype=dtype)
        else:
            print(f"  BRIDGE dimensions match ({hidden_size}), preserving weights")
            self.bridge = self.bridge.to(target_device, dtype=dtype)
        
        # VTO head - check input dimension
        vto_input_dim = None
        if hasattr(self.vto_head, '__getitem__') or hasattr(self.vto_head, '__iter__'):
            try:
                first_layer = self.vto_head[0]
                if hasattr(first_layer, 'in_features'):
                    vto_input_dim = first_layer.in_features
            except:
                pass
        
        if vto_input_dim is None or vto_input_dim != hidden_size:
            print(f"  Reinitializing vto_head: {vto_input_dim} -> {hidden_size}")
            self.vto_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_vtos)
            ).to(target_device, dtype=dtype)
        else:
            print(f"  vto_head dimensions match ({hidden_size}), preserving weights")
            self.vto_head = self.vto_head.to(target_device, dtype=dtype)
        
        # Recommendation head - check input dimension
        rec_input_dim = None
        if hasattr(self.recommendation_head, '__getitem__') or hasattr(self.recommendation_head, '__iter__'):
            try:
                first_layer = self.recommendation_head[0]
                if hasattr(first_layer, 'in_features'):
                    rec_input_dim = first_layer.in_features
            except:
                pass
        
        if rec_input_dim is None or rec_input_dim != hidden_size:
            print(f"  Reinitializing recommendation_head: {rec_input_dim} -> {hidden_size}")
            self.recommendation_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, 1)
            ).to(target_device, dtype=dtype)
        else:
            print(f"  recommendation_head dimensions match ({hidden_size}), preserving weights")
            self.recommendation_head = self.recommendation_head.to(target_device, dtype=dtype)

        # Retrieval modules were left out of both the resize and the device move,
        # so they kept ModelConfig's default 3584 regardless of the backbone.
        if getattr(self, "retriever", None) is not None:
            cfg = self.retrieval_config
            if self.retriever.hidden_size != hidden_size:
                print(f"  Reinitializing retriever: {self.retriever.hidden_size} -> {hidden_size}")
                self.retriever = TwoTowerRetriever(
                    hidden_size, embed_dim=cfg.embed_dim, dropout=cfg.dropout,
                    init_temperature=cfg.temperature)
            # float32 regardless of backbone dtype; see load_base_model.
            self.retriever = self.retriever.to(target_device, dtype=torch.float32)
            if self.item_id_embedding is not None:
                self.item_id_embedding = self.item_id_embedding.to(target_device, dtype=torch.float32)
            if getattr(self, "item_bias", None) is not None:
                self.item_bias = self.item_bias.to(target_device, dtype=torch.float32)
            if self.reranker is not None:
                if self.reranker.scorer[0].in_features != hidden_size * 4:
                    self.reranker = CrossEncoderReranker(hidden_size, cfg.dropout)
                self.reranker = self.reranker.to(target_device, dtype=torch.float32)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_for_sft(self):
        """Enable gradients on the modules trained during SFT.

        base_model (LoRA adapters), vto_head, recommendation_head, BRIDGE
        and the retrieval heads.
        """
        for param in self.vto_head.parameters():
            param.requires_grad = True
        for param in self.recommendation_head.parameters():
            param.requires_grad = True
        for param in self.bridge.parameters():
            param.requires_grad = True

        for _mod in (self.retriever, self.item_id_embedding,
                     getattr(self, "item_bias", None), self.reranker):
            if _mod is not None:
                for param in _mod.parameters():
                    param.requires_grad = True

        print("✓ Training: base_model + vto_head + recommendation_head + BRIDGE")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                domain: Union[Domain, torch.Tensor] = Domain.GENERAL,
                vto_labels: Optional[torch.Tensor] = None,
                mode: str = "full",
                training_stage: str = None,
                context_mask: Optional[torch.Tensor] = None,
                **kwargs) -> ModelOutput:
        """The supervised fine-tuning pass (``training_stage="sft"``), or with
        ``mode="hidden_states"`` only the mean-pooled last-layer states.

        The SFT loss is the LM loss plus the VTO and recommendation heads and
        BRIDGE's auxiliary losses; ``context_pooled`` (the mean state of the
        dialogue tokens, ``context_mask``) is what the retriever encodes.
        """
        if self.base_model is None:
            raise RuntimeError("Call load_base_model() first")
        
        # Pooled hidden states only (item encoding for the retrieval objectives)
        if mode == "hidden_states":
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                logits_to_keep=1,  # hidden states only; skip full-vocab logits
                **kwargs
            )
            # Return pooled hidden states (BRIDGE is applied explicitly in training code)
            return masked_mean_pool(outputs.hidden_states[-1], attention_mask)
        
        if training_stage != "sft":
            raise ValueError(f"training_stage={training_stage!r}: only 'sft' is part of this pipeline")

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs
        )
        
        hidden_states = outputs.hidden_states[-1]
        device = input_ids.device

        # Retrieval context: raw last-layer states over the dialogue tokens only.
        # The SFT sequence also holds the reply, which names the target item in
        # every ReDial example, and evaluation encodes the dialogue without
        # BRIDGE. Pooling the BRIDGE features of the whole sequence trained the
        # retriever to spot the answer in text it never sees at test time.
        context_pooled = (masked_mean_pool(hidden_states, context_mask)
                          if context_mask is not None else None)
        
        # Initialize outputs
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        # BRIDGE features, then the VTO and recommendation heads
        bridge_out = self.bridge(hidden_states, domain, vto_labels=vto_labels,
                                 attention_mask=attention_mask)
        adapted_hidden = bridge_out["features"]
        
        # Pool adapted hidden states
        pooled = masked_mean_pool(adapted_hidden, attention_mask)
        
        # VTO prediction
        vto_logits = self.vto_head(pooled)
        
        # Recommendation head training (self-supervised via LM confidence)
        rec_scores = self.recommendation_head(pooled)
        
        # VTO loss
        if vto_labels is not None:
            vto_loss = F.binary_cross_entropy_with_logits(vto_logits, vto_labels.float())
            total_loss = total_loss + 0.3 * vto_loss
        
        # BRIDGE domain confusion loss
        if bridge_out.get("domain_loss") is not None:
            total_loss = total_loss + 0.1 * bridge_out["domain_loss"]
        
        # BRIDGE task preservation loss
        if bridge_out.get("task_loss") is not None:
            total_loss = total_loss + 0.1 * bridge_out["task_loss"]
        
        # NEW: BRIDGE contrastive loss for cross-domain alignment
        if bridge_out.get("contrastive_loss") is not None:
            total_loss = total_loss + 0.05 * bridge_out["contrastive_loss"]
        
        # Self-supervised recommendation head loss
        # Use LM confidence as target (higher confidence = better recommendation quality)
        # outputs.loss is the batch-mean LM loss: expanding it gave every
        # example the same target, so the head could only learn a constant.
        if labels is not None and outputs.logits is not None:
            rec_target = torch.exp(-self._per_example_nll(outputs.logits, labels))
            rec_target = rec_target.clamp(0.1, 0.9).detach()
            rec_loss = F.mse_loss(torch.sigmoid(rec_scores).squeeze(-1), rec_target)
            total_loss = total_loss + 0.1 * rec_loss

        return ModelOutput(
            logits=outputs.logits,
            loss=total_loss,
            hidden_states=adapted_hidden,
            vto_logits=vto_logits,
            context_pooled=context_pooled
        )
