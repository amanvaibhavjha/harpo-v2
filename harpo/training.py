"""
HARPO: 4-Stage Training Pipeline

Optimizing Conversational Recommendation for User-Aligned Quality
via Hierarchical Preference Learning

Stages:
1. SFT: Supervised fine-tuning with VTO prediction (λ_v weighted)
2. CHARM: Hierarchical preference optimization  
3. STAR: Tree-of-thought reasoning training with value network
4. MAVEN: Multi-agent self-play refinement

Primary Objective: USER-ALIGNED RECOMMENDATION QUALITY
"""

import os
import random
from typing import Dict, List, Optional

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CRITICAL FIX: Disable torch dynamo/compile to avoid conflicts with DDP
# This prevents "Unsupported method call Logger.set_runtime_stats_and_log" error
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Disable dynamo after torch import as well
torch._dynamo.config.suppress_errors = True
try:
    torch._dynamo.disable()
except:
    pass

from .config import VTO, Domain, TrainingConfig
from .model import HARPOMTv2
from .pooling import masked_mean_pool
from .retrieval import CatalogSoftmax, NegativeQueue


def token_subset_ce(logits: torch.Tensor, labels: torch.Tensor,
                    token_ids: List[int]) -> torch.Tensor:
    """Cross-entropy restricted to positions whose *target* is in ``token_ids``.

    The causal shift matters and was missing: logits[:, t] predicts token t+1, so
    the original flattened pairing of logits[t] with labels[t] trained the model
    to emit each special token one position early, at weight 0.5.
    """
    if not token_ids:
        return logits.new_zeros(())
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    wanted = torch.zeros_like(shift_labels, dtype=torch.bool)
    for tid in token_ids:
        wanted |= shift_labels == tid
    wanted &= shift_labels != -100
    if not bool(wanted.any()):
        return logits.new_zeros(())
    return F.cross_entropy(shift_logits[wanted], shift_labels[wanted])



# ============================================================================
# DATASETS
# ============================================================================

class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning
    
    CRITICAL FIXES APPLIED:
    1. Disabled chat template - our custom format must match inference exactly
    2. Fixed label masking to use consistent tokenization approach
    3. ADDED: VTO content emphasis to ensure model learns VTO sequences
    
    The training format is:
        input_text + "\\nAssistant: " + output_text + eos_token
    
    This MUST match what we use during generation in evaluation.py!
    
    VTO Content Learning:
    The model must learn that after <|think|>, it should output VTO names
    like "extract_context, extract_entities" instead of gibberish.
    We achieve this by:
    1. Not masking any part of the output (full supervision)
    2. Using proper tokenization that preserves the VTO pattern
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512,
                 item_index: Optional[Dict[str, int]] = None,
                 item_max_length: int = 48):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vto_to_idx = {vto.value: i for i, vto in enumerate(VTO)}
        self.num_vtos = len(VTO)
        # Catalogue index for the retriever's positive. Without it the retrieval
        # objective is skipped and behaviour is unchanged for existing callers.
        self.item_index = item_index or {}
        self.item_max_length = item_max_length
        
        # ===== CRITICAL FIX #1: DISABLE CHAT TEMPLATE =====
        # Your data already has custom format like:
        #   Input: "<|domain:books|>\n\nUser: I'm looking for comic books"
        #   Output: "<|think|>extract_context<|/think|>\nGreat! I can help..."
        # 
        # Qwen's chat template would convert this to:
        #   <|im_start|>system....<|im_end|><|im_start|>user....
        # 
        # But during INFERENCE (evaluation.py), you DON'T use chat template!
        # This mismatch causes the model to output gibberish.
        #
        # SOLUTION: Always use simple format, no chat template
        self.use_chat_template = False  # FORCE DISABLED
        
        # Verify tokenizer has special tokens
        self._verify_special_tokens()
    
    def _verify_special_tokens(self):
        """Verify tokenizer has all required special tokens"""
        required = ["<|think|>", "<|/think|>", "<|tool_start|>", "<|tool_end|>"]
        vocab = self.tokenizer.get_vocab()
        for token in required:
            if token not in vocab:
                print(f"⚠️ Warning: Token '{token}' not in tokenizer vocabulary!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item["input"]
        output_text = item["output"]
        
        # ===== CRITICAL FIX #2: CONSISTENT FORMAT =====
        # This format MUST EXACTLY MATCH what we use in evaluation.py generate!
        # Format: input_text + "\nAssistant: " + output_text + eos
        input_portion = input_text + "\nAssistant: "
        full_text = input_portion + output_text + self.tokenizer.eos_token
        
        # OPTIMIZATION: Single tokenization with offset mapping to find split point
        # This replaces the previous double-tokenization approach
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True  # Get character offsets
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)
        
        # OPTIMIZATION: Find input_len using offset mapping instead of re-tokenizing
        # Find the first token that starts at or after len(input_portion)
        input_portion_len = len(input_portion)
        input_len = 0
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start >= input_portion_len:
                input_len = i
                break
            input_len = i + 1  # Include this token if it overlaps
        
        # Create labels - mask input portion (only train on output)
        labels = input_ids.clone()
        labels[:min(input_len, self.max_length)] = -100
        labels[attention_mask == 0] = -100  # Also mask padding
        
        # VTO labels
        vto_labels = torch.zeros(self.num_vtos)
        for vto_name in item.get("vtos", []):
            if vto_name in self.vto_to_idx:
                vto_labels[self.vto_to_idx[vto_name]] = 1.0
        
        # Domain
        domain_str = item.get("domain", "general")
        domain_idx = 3  # Default to general
        for i, d in enumerate(Domain):
            if d.value == domain_str:
                domain_idx = i
                break
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vto_labels": vto_labels,
            "domain_idx": domain_idx,
        }

        gt_item = item.get("ground_truth_item")
        if gt_item:
            item_text = str(gt_item)
            meta = item.get("item_metadata") or {}
            for key in ("year", "genre", "director", "category", "brand"):
                if meta.get(key):
                    item_text += f" | {key}: {meta[key]}"
            enc = self.tokenizer(item_text, max_length=self.item_max_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
            batch["item_input_ids"] = enc["input_ids"].squeeze(0)
            batch["item_attention_mask"] = enc["attention_mask"].squeeze(0)
            # -1 means "not in the catalogue": encode_item_embeddings falls back
            # to content-only rather than indexing out of range.
            batch["item_index"] = self.item_index.get(str(gt_item).lower(), -1)
            batch["has_item"] = 1
        else:
            pad = torch.zeros(self.item_max_length, dtype=torch.long)
            batch["item_input_ids"] = pad
            batch["item_attention_mask"] = pad.clone()
            batch["item_index"] = -1
            batch["has_item"] = 0

        return batch


def trim_padding_collate(batch):
    """Collate, then drop columns that are padding in every row of the batch.

    SFTDataset pads to ``max_length``, so a batch of short dialogues still paid
    for a full-length forward; on ReDial (median ~190 of 256 tokens) about a
    quarter of all compute was padding.
    """
    out = default_collate(batch)
    for mask_key, keys in (("attention_mask", ("input_ids", "attention_mask", "labels")),
                           ("item_attention_mask", ("item_input_ids", "item_attention_mask"))):
        if mask_key not in out:
            continue
        live = out[mask_key].bool().any(dim=0).nonzero().flatten()
        if live.numel() == 0:
            continue
        start, end = int(live[0]), int(live[-1]) + 1
        for key in keys:
            if key in out:
                out[key] = out[key][:, start:end]
    return out


# ============================================================================
# TRAINER
# ============================================================================

class HARPOMTv2Trainer:
    """Trains the retriever (supervised fine-tuning with the full-catalogue
    objective) on one device, and saves / loads its checkpoints."""

    # Saved and restored with the other components; absent ones are skipped.
    RETRIEVAL_COMPONENTS = ("retriever", "item_id_embedding", "item_bias", "reranker")

    def __init__(self, model: HARPOMTv2, config: TrainingConfig, device: str = None):
        self.model = model
        self.config = config
        self._catalog_softmax = None   # set by attach_catalog()
        self._catalog_refresh_every = 400
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"Trainer using device: {self.device}")

        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.eval_metrics = []
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _get_base_model(self):
        return self.model

    def attach_catalog(self, item_texts, item_index, refresh_every: int = 400,
                       item_max_length: int = 32, encode_batch_size: int = 64,
                       item_counts=None):
        """Enable the full-catalogue softmax objective for Stage 1.

        Without it the retrieval objective only ever sees in-batch negatives --
        3 at batch 4 -- while evaluation ranks against thousands. That gap is why
        in-batch accuracy reached 0.66 while full-catalogue R@10 sat at chance.
        """
        base_model = self._get_base_model()
        if getattr(base_model, "retriever", None) is None:
            print("  Catalogue objective skipped: retrieval is disabled")
            return
        self._catalog_texts = list(item_texts)
        self._catalog_refresh_every = max(1, refresh_every)
        self._catalog_item_max_length = item_max_length
        self._catalog_encode_batch = encode_batch_size
        self._catalog_softmax = CatalogSoftmax(
            base_model.retriever, base_model.item_id_embedding,
            num_items=len(self._catalog_texts),
            id_weight=base_model.retrieval_config.id_embedding_weight,
            item_bias=getattr(base_model, "item_bias", None),
        ).to(self.device)
        print(f"  Catalogue objective ON: softmax over {len(self._catalog_texts)} "
              f"items, content refreshed every {self._catalog_refresh_every} steps")

        bias = getattr(base_model, "item_bias", None)
        if bias is not None and item_counts is not None:
            # log(count + 1): the softmax of the bias alone is the (add-one
            # smoothed) training popularity distribution.
            counts = torch.as_tensor(list(item_counts), dtype=torch.float32)
            with torch.no_grad():
                bias.weight.zero_()
                bias.weight[:len(counts), 0] = torch.log1p(counts).to(bias.weight.device)
            print(f"  Item bias initialised to log popularity "
                  f"({int((counts > 0).sum())}/{len(counts)} items seen in training)")

    @torch.no_grad()
    def _refresh_catalog_content(self):
        """Re-encode item text through the backbone into the cached matrix."""
        base_model = self._get_base_model()
        was_training = base_model.training
        base_model.eval()
        try:
            chunks, step = [], self._catalog_encode_batch
            for start in range(0, len(self._catalog_texts), step):
                enc = base_model.tokenizer(
                    self._catalog_texts[start:start + step], return_tensors="pt",
                    padding=True, truncation=True,
                    max_length=self._catalog_item_max_length).to(self.device)
                out = base_model.base_model(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                    output_hidden_states=True, logits_to_keep=1)
                pooled = masked_mean_pool(out.hidden_states[-1], enc["attention_mask"])
                chunks.append(base_model.retriever.encode_item(pooled))
            self._catalog_softmax.set_content(torch.cat(chunks, dim=0))
        finally:
            base_model.train(was_training)

    def _setup_optimizer(self, num_steps: int, lr: float):
        base_model = self._get_base_model()
        
        # OPTIMIZATION: Use fused AdamW for ~10-15% speedup on CUDA
        optimizer_kwargs = {
            "lr": lr,
            "weight_decay": self.config.weight_decay,
        }
        # Fused optimizer only available on CUDA
        if str(self.device).startswith("cuda"):
            optimizer_kwargs["fused"] = True
        
        self.optimizer = AdamW(
            [p for p in base_model.parameters() if p.requires_grad],
            **optimizer_kwargs
        )
        
        warmup_steps = int(num_steps * self.config.warmup_ratio)
        
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_steps)
        decay = CosineAnnealingLR(self.optimizer, T_max=num_steps - warmup_steps)
        
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, decay], milestones=[warmup_steps]
        )
    
    def _create_dataloader(self, dataset, shuffle=True, drop_last=True):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=4,
            # Pinning is CUDA-only; on MPS it only emits a warning.
            pin_memory=str(self.device).startswith('cuda'),
            persistent_workers=True,
            drop_last=drop_last,
            collate_fn=trim_padding_collate,
        )

    def _get_amp_context(self):
        """Full precision: training runs without autocast."""
        return torch.amp.autocast('cuda', enabled=False)

    def _optimizer_step(self):
        base_model = self._get_base_model()
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def train_sft(self, train_dataset: SFTDataset, epoch_end_callback=None) -> List[Dict]:
        """Stage 1: Supervised Fine-Tuning
        
        CRITICAL FIX: Added special token loss AND VTO content loss to ensure 
        model learns to generate proper <|think|>VTO_names<|/think|> format.
        
        The key insight: The model must learn TWO things:
        1. When to output special tokens like <|think|>
        2. What content to output between the special tokens (VTO names)
        
        Without explicit VTO content supervision, the model may output <|think|>
        but then generate gibberish instead of VTO names.
        """
        print("\n" + "=" * 60)
        print("Stage 1: Supervised Fine-Tuning (SFT)")
        print("=" * 60)
        
        # ===== CRITICAL FIX: Freeze unused modules for DDP compatibility =====
        # This prevents "parameters not used in loss" errors
        base_model = self._get_base_model()
        base_model.freeze_for_sft()
        
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        
        actual_batch_size = train_loader.batch_size
        grad_accum = self.config.gradient_accumulation_steps
        
        num_steps = len(train_loader) * self.config.sft_epochs // grad_accum
        self._setup_optimizer(num_steps, self._get_base_model().model_config.sft_learning_rate)
        
        print(f"Training config:")
        print(f"  Batch size: {actual_batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {actual_batch_size * grad_accum}")
        print(f"  Total steps: {num_steps}")
        
        # CRITICAL FIX: Get special token IDs for auxiliary loss
        base_model = self._get_base_model()
        tokenizer = base_model.tokenizer
        # These losses ask the model to emit specific token ids, which is only
        # achievable if embed_tokens/lm_head are trainable. Under LoRA alone they
        # are frozen, so the objective has an unreachable optimum: measured on
        # Qwen2.5-0.5B it starts at 17.6 nats and plateaus at 8.3, weighted 0.5 --
        # a large permanent gradient that diverged epoch 1 into NaN.
        embeddings_trainable = any(
            p.requires_grad for n, p in base_model.base_model.named_parameters()
            if "embed_tokens" in n or "lm_head" in n
        )
        if not embeddings_trainable:
            print("  Token-level aux losses DISABLED: embed_tokens/lm_head frozen, "
                  "so special-token targets are unreachable")

        special_token_ids = set()
        for token in ["<|think|>", "<|/think|>", "<|tool_start|>", "<|tool_end|>"]:
            ids = tokenizer.encode(token, add_special_tokens=False)
            special_token_ids.update(ids)
        special_token_ids = list(special_token_ids) if embeddings_trainable else []
        print(f"  Special tokens to emphasize: {len(special_token_ids)} token IDs")
        
        # NEW: Get VTO name token IDs for additional supervision
        vto_names = ["extract_context", "extract_entities", "compare_options", "refine_query",
                     "retrieve_preferences", "identify_constraints", "search_candidates",
                     "rank_options", "filter_results", "explain_choice"]
        vto_token_ids = set()
        for vto_name in vto_names:
            ids = tokenizer.encode(vto_name, add_special_tokens=False)
            vto_token_ids.update(ids)
        vto_token_ids = list(vto_token_ids) if embeddings_trainable else []
        print(f"  VTO content tokens to emphasize: {len(vto_token_ids)} token IDs")
        
        retrieval_cfg = getattr(self.config, "retrieval_config", None)
        retriever_enabled = (retrieval_cfg is not None and retrieval_cfg.enabled
                             and getattr(base_model, "retriever", None) is not None)
        retrieval_weight = retrieval_cfg.loss_weight if retrieval_cfg else 0.0

        # Cross-batch negatives: at batch 4 the in-batch objective sees only 3
        # negatives against a catalogue of thousands.
        negative_queue = None
        if retriever_enabled and retrieval_cfg.num_hard_negatives > 0:
            negative_queue = NegativeQueue(
                embed_dim=retrieval_cfg.embed_dim,
                capacity=max(1024, 128 * retrieval_cfg.num_hard_negatives))
        if retriever_enabled:
            print(f"  Retrieval objective ON (weight {retrieval_weight}, "
                  f"item repr '{retrieval_cfg.item_representation}')")

        catalog_softmax = self._catalog_softmax
        catalog_ready = False

        self.model.train()
        global_step = 0

        for epoch in range(self.config.sft_epochs):
            # On-device accumulators reduced once per epoch. Five .item() calls
            # per step drained the MPS queue and held utilisation near 2%.
            _dev = self.device
            epoch_loss = torch.zeros((), device=_dev)
            epoch_special_loss = torch.zeros((), device=_dev)
            epoch_vto_content_loss = torch.zeros((), device=_dev)
            epoch_retrieval_acc = torch.zeros((), device=_dev)
            epoch_catalog_rank = torch.zeros((), device=_dev)
            epoch_catalog_acc = torch.zeros((), device=_dev)
            num_batches = 0
            nan_steps = 0
            catalog_batches = 0
            
            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{self.config.sft_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                
                domain_idx = batch["domain_idx"][0].item()
                domain = list(Domain)[domain_idx]
                
                with self._get_amp_context():
                    # CRITICAL: Pass training_stage="sft" for DDP compatibility
                    # This ensures only modules that contribute to loss are called
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        domain=domain,
                        vto_labels=batch["vto_labels"],
                        training_stage="sft",  # Only use base_model + vto_head
                        # Dialogue tokens only (labels mask the prompt), so the
                        # retriever sees what evaluation sees.
                        context_mask=batch["attention_mask"] * (batch["labels"] == -100)
                    )
                    
                    base_loss = outputs.loss
                    
                    # CRITICAL FIX: Add special token emphasis loss
                    # This helps the model learn when to generate special tokens
                    # Use batch device for DDP compatibility
                    batch_device = batch["input_ids"].device
                    special_loss = torch.tensor(0.0, device=batch_device, dtype=base_loss.dtype)
                    vto_content_loss = torch.tensor(0.0, device=batch_device, dtype=base_loss.dtype)
                    
                    if outputs.logits is not None:
                        # Causally shifted inside token_subset_ce; the previous
                        # inline version compared logits[t] with labels[t].
                        special_loss = token_subset_ce(
                            outputs.logits, batch["labels"], special_token_ids)
                        vto_content_loss = token_subset_ce(
                            outputs.logits, batch["labels"], vto_token_ids)

                    # ===== Retrieval objective (in-batch + queue) =====
                    retrieval_loss = torch.tensor(0.0, device=batch_device,
                                                  dtype=base_loss.dtype)
                    retrieval_acc = torch.zeros((), device=batch_device)
                    if retriever_enabled and int(batch.get("has_item", torch.zeros(1)).sum()) >= 2:
                        keep = batch["has_item"].bool()
                        item_hidden = self.model(
                            input_ids=batch["item_input_ids"][keep],
                            attention_mask=batch["item_attention_mask"][keep],
                            mode="hidden_states")
                        ctx_hidden = outputs.context_pooled[keep]
                        idx = batch["item_index"][keep]

                        item_embeds = base_model.encode_item_embeddings(item_hidden, idx)
                        ctx_embeds = base_model.retriever.encode_context(ctx_hidden)
                        logits = (ctx_embeds @ item_embeds.t()) / base_model.retriever.temperature

                        # Mask duplicate positives: a few titles dominate ReDial,
                        # and otherwise a repeated item is pushed away from itself.
                        dup = (idx.unsqueeze(0) == idx.unsqueeze(1)) & (idx.unsqueeze(0) >= 0)
                        dup &= ~torch.eye(dup.size(0), dtype=torch.bool, device=dup.device)
                        logits = logits.masked_fill(dup, torch.finfo(logits.dtype).min)

                        if (negative_queue is not None
                                and epoch >= retrieval_cfg.hard_negative_start_epoch
                                and len(negative_queue) > 0):
                            hard = negative_queue.sample_hard(
                                ctx_embeds, retrieval_cfg.num_hard_negatives,
                                exclude=idx, debias=retrieval_cfg.popularity_debias)
                            if hard is not None:
                                hl = torch.einsum("bd,bnd->bn", ctx_embeds,
                                                  hard.to(ctx_embeds.dtype))
                                logits = torch.cat(
                                    [logits, hl / base_model.retriever.temperature], dim=1)

                        targets = torch.arange(logits.size(0), device=logits.device)
                        retrieval_loss = F.cross_entropy(logits, targets)
                        retrieval_acc = (logits.argmax(1) == targets).float().mean().detach()
                        if negative_queue is not None:
                            negative_queue.enqueue(item_embeds, idx)

                    # ===== Full-catalogue softmax: the objective the metric measures
                    catalog_loss = torch.tensor(0.0, device=batch_device,
                                                dtype=base_loss.dtype)
                    if catalog_softmax is not None:
                        if (not catalog_ready
                                or num_batches % self._catalog_refresh_every == 0):
                            self._refresh_catalog_content()
                            catalog_ready = True
                        cat_out = catalog_softmax(
                            outputs.context_pooled, batch["item_index"].to(batch_device))
                        catalog_loss = cat_out["loss"]
                        epoch_catalog_rank = epoch_catalog_rank + cat_out["rank"]
                        epoch_catalog_acc = epoch_catalog_acc + cat_out["accuracy"]
                        catalog_batches += 1

                    # Total loss with special token AND VTO content emphasis
                    total_loss = (base_loss + 0.5 * special_loss
                                  + 0.3 * vto_content_loss
                                  + retrieval_weight * retrieval_loss
                                  + retrieval_weight * catalog_loss)

                    # Attribute non-finite losses rather than averaging a NaN into
                    # the epoch and reporting only "Avg Loss: nan".
                    if not torch.isfinite(total_loss):
                        parts = {"lm": base_loss, "special": special_loss,
                                 "vto": vto_content_loss,
                                 "retrieval": retrieval_loss, "catalog": catalog_loss}
                        bad = [k for k, v in parts.items() if not torch.isfinite(v)]
                        print(f"\n  !! non-finite loss at epoch {epoch} step "
                              f"{batch_idx}: {bad or ['sum']}")
                        nan_steps += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        continue
                
                loss = total_loss / grad_accum
                loss.backward()
                
                if (batch_idx + 1) % grad_accum == 0:
                    self._optimizer_step()
                    global_step += 1
                
                epoch_loss = epoch_loss + base_loss.detach()
                epoch_special_loss = epoch_special_loss + special_loss.detach()
                epoch_vto_content_loss = epoch_vto_content_loss + vto_content_loss.detach()
                epoch_retrieval_acc = epoch_retrieval_acc + retrieval_acc
                num_batches += 1
                if num_batches % 50 == 0:
                    pbar.set_postfix({"loss": f"{epoch_loss.item()/num_batches:.3f}"})
            
            avg_loss = epoch_loss.item() / max(num_batches, 1)
            avg_special = epoch_special_loss.item() / max(num_batches, 1)
            avg_vto_content = epoch_vto_content_loss.item() / max(num_batches, 1)
            self.train_losses.append({
                "epoch": epoch, "stage": "sft", "loss": avg_loss,
                "special_loss": avg_special, "vto_content_loss": avg_vto_content,
                "retrieval_acc": epoch_retrieval_acc.item() / max(num_batches, 1),
                "catalog_rank": (epoch_catalog_rank.item() / catalog_batches
                                 if catalog_batches else None),
                "catalog_acc": (epoch_catalog_acc.item() / catalog_batches
                                if catalog_batches else None),
                "nan_steps": nan_steps,
            })
            if catalog_batches:
                print(f"  catalogue: mean rank "
                      f"{epoch_catalog_rank.item()/catalog_batches:.1f} / "
                      f"{catalog_softmax.num_items}, top-1 "
                      f"{100*epoch_catalog_acc.item()/catalog_batches:.2f}%")
            print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Special Token Loss: {avg_special:.4f}, VTO Content Loss: {avg_vto_content:.4f}")
            
            # e.g. a full ranking evaluation, so results exist per epoch even if
            # a long run is cut short.
            if epoch_end_callback is not None:
                epoch_end_callback(epoch, self.train_losses[-1])
        
        self.save_checkpoint("sft_final")
        
        return self.train_losses
    
    def save_checkpoint(self, name: str):
        """Save the component weights, the LoRA adapter and the tokenizer.

        CRITICAL FIXES:
        1. Save tokenizer with new special tokens
        2. Verify and save modules_to_save (embed_tokens, lm_head) explicitly
        
        The modules_to_save are critical for special token generation!
        Without them, <|think|>, <|/think|> etc. would have random embeddings.
        """
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints", name)
        print(f"Creating checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        base_model = self._get_base_model()
        
        # Save components
        components_path = os.path.join(checkpoint_dir, "components.pt")
        torch.save({
            "bridge_state_dict": base_model.bridge.state_dict(),
            "vto_head_state_dict": base_model.vto_head.state_dict(),
            "recommendation_head_state_dict": base_model.recommendation_head.state_dict(),
            # The retrieval heads produce every ranking score; a checkpoint
            # without them cannot be evaluated.
            **{f"{name}_state_dict": module.state_dict()
               for name in self.RETRIEVAL_COMPONENTS
               if (module := getattr(base_model, name, None)) is not None},
            "train_losses": self.train_losses,
            "eval_metrics": self.eval_metrics
        }, components_path)
        print(f"✓ Saved components to: {components_path}")
        
        # Save LoRA adapter (this should include modules_to_save)
        base_model_dir = os.path.join(checkpoint_dir, "base_model")
        if hasattr(base_model.base_model, 'save_pretrained'):
            base_model.base_model.save_pretrained(base_model_dir)
            print(f"✓ Saved LoRA adapter to: {base_model_dir}")
            
            # VERIFY modules_to_save are in the checkpoint
            #import os
            adapter_files = os.listdir(base_model_dir)
            print(f"  Adapter files: {adapter_files}")
            
            # Check if modules_to_save weights are in the safetensors file
            safetensors_path = os.path.join(base_model_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                try:
                    from safetensors import safe_open
                    with safe_open(safetensors_path, framework="pt") as f:
                        keys = list(f.keys())
                        embed_keys = [k for k in keys if "embed" in k.lower()]
                        lm_head_keys = [k for k in keys if "lm_head" in k.lower()]
                        print(f"  Embed token weights in checkpoint: {len(embed_keys) > 0}")
                        print(f"  LM head weights in checkpoint: {len(lm_head_keys) > 0}")
                        if embed_keys:
                            print(f"    Embed keys: {embed_keys[:3]}...")
                        if lm_head_keys:
                            print(f"    LM head keys: {lm_head_keys[:3]}...")
                except Exception as e:
                    print(f"  Could not verify safetensors contents: {e}")
        
        # ===== CRITICAL FIX: Save tokenizer with special tokens! =====
        if base_model.tokenizer is not None:
            base_model.tokenizer.save_pretrained(base_model_dir)
            print(f"✓ Saved tokenizer with {len(base_model.tokenizer)} tokens")
            
            # Verify special tokens are in the tokenizer
            special_tokens = ["<|think|>", "<|/think|>", "<|tool_start|>", "<|tool_end|>"]
            vocab = base_model.tokenizer.get_vocab()
            for token in special_tokens:
                if token in vocab:
                    print(f"  ✓ '{token}' in vocab (id: {vocab[token]})")
                else:
                    print(f"  ⚠ '{token}' NOT in vocab!")
        
        print(f"✓ Checkpoint saved: {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint with trained component weights

        This loads bridge, vto_head, recommendation_head weights from
        components.pt. The LoRA adapter is loaded separately in load_base_model.
        
        Args:
            checkpoint_path: Path to checkpoint directory (e.g., 'outputs/checkpoints/charm_final')
        """
        base_model = self._get_base_model()
        
        components_path = os.path.join(checkpoint_path, "components.pt")
        if not os.path.exists(components_path):
            print(f"⚠ Warning: components.pt not found at {components_path}")
            print("  Only LoRA adapter will be loaded, auxiliary modules will be randomly initialized")
            return
        
        print(f"Loading component weights from: {components_path}")
        checkpoint = torch.load(components_path, map_location=self.device)
        
        # Load each component with error handling
        components = [
            ("bridge", "bridge_state_dict"),
            ("vto_head", "vto_head_state_dict"),
            ("recommendation_head", "recommendation_head_state_dict"),
        ] + [(name, f"{name}_state_dict") for name in self.RETRIEVAL_COMPONENTS
             if getattr(base_model, name, None) is not None]
        
        for component_name, state_dict_key in components:
            if state_dict_key in checkpoint:
                component = getattr(base_model, component_name, None)
                if component is not None:
                    try:
                        component.load_state_dict(checkpoint[state_dict_key])
                        print(f"  ✓ Loaded {component_name}")
                    except Exception as e:
                        print(f"  ⚠ Failed to load {component_name}: {e}")
                else:
                    print(f"  ⚠ Component {component_name} not found in model")
            else:
                print(f"  ⚠ {state_dict_key} not found in checkpoint")
        
        self.train_losses = checkpoint.get("train_losses", [])
        self.eval_metrics = checkpoint.get("eval_metrics", [])
        
        print(f"Checkpoint loaded: {checkpoint_path}")


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
