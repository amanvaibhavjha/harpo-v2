"""
CHARM: a cross-encoder re-ranker. The LLM reads the dialogue and one candidate
together and scores it; unlike the two-tower retriever, every candidate token
attends to every dialogue token.

  * It starts from the plain instruct model with a fresh LoRA adapter, not from
    the retriever's fine-tuned backbone, which has memorised the training answers.
  * It never sees the retriever's score, so a missed positive inserted into a
    training group is harmless.
  * Dimensions (``heads``): relevance (is this what the recommender suggests
    now), satisfaction (will the seeker like it; ReDial's own "liked" answers),
    engagement (will the seeker take it up; the next seeker turns) and,
    optionally, diversity (``heads=4``; dissimilarity to what the dialogue has
    already discussed -- a per-candidate proxy, see harpo/diversity.py's
    ``dialogue_diversity_target``, for a property that is really the whole
    list's). A gate reads the dialogue and mixes them per conversation:
    ``score = sum_d w_d(dialogue) * s_d``. ``heads=1`` is relevance alone
    (training stage 1). The post-hoc, list-level diversity re-ranking
    (``mmr_rerank``) in harpo/diversity.py is separate and unaffected by this.
  * Inputs: a preference reading appended to the dialogue ("Seeker wants: ..."),
    which is also how STAR's search steers CHARM; optionally a BRIDGE profile
    appended to each candidate.

At evaluation its score is fused with the retriever's, the weight chosen on
validation (weight 0 -- the retriever alone -- is always a candidate).
"""

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

SUFFIX = '\nAssistant: How about "{title}"?'
# Token budget reserved for the candidate suffix. Fixing it (rather than sizing
# the dialogue to each title) makes the dialogue prefix identical for every
# candidate, which is what lets them share one encoding of it.
CAND_BUDGET = 32
PROFILE_BUDGET = 64          # candidate budget when BRIDGE profiles are appended
SUMMARY = "\nSeeker wants: {summary}"
SUMMARY_BUDGET = 48


def sample_group(positive: int, candidates: Sequence[int], exclude: Sequence[int],
                 group: int, num_items: int, rng: random.Random,
                 strata: Optional[Sequence[Tuple[int, float]]] = None) -> List[int]:
    """``[positive] + (group - 1)`` distinct negatives, hard ones first.

    Negatives come from the retriever's shortlist minus ``exclude`` (every
    positive of the turn, or of the whole conversation); if the shortlist runs
    short, random catalogue items fill the rest.

    ``strata`` -- e.g. ``[(50, 0.7), (200, 0.25)]`` -- draws that share of the
    negatives from shortlist ranks below each bound (ranks 0-49, then 50-199);
    the remainder are random catalogue items, so the model also learns to push
    obscure titles down. Without it, negatives are uniform over the shortlist.
    """
    banned = set(exclude) | {positive}
    hard = [c for c in dict.fromkeys(candidates) if c not in banned]
    need = group - 1
    if strata:
        negs, lo, pool = [], 0, list(dict.fromkeys(candidates))
        for hi, share in strata:
            band = [c for c in pool[lo:hi] if c not in banned and c not in negs]
            k = min(len(band), int(round(share * need)), need - len(negs))
            negs += rng.sample(band, k)
            lo = hi
    else:
        negs = rng.sample(hard, need) if len(hard) >= need else list(hard)
    taken = banned | set(negs)
    while len(negs) < need:
        c = rng.randrange(num_items)
        if c not in taken:
            negs.append(c)
            taken.add(c)
    return [positive] + negs


def fused_ranks(retriever: torch.Tensor, other: torch.Tensor, weight: float,
                target_pos: torch.Tensor, retriever_rank: torch.Tensor) -> torch.Tensor:
    """1-based target ranks under ``(1-w) z(retriever) + w z(other)`` within each list.

    Rows whose target is outside the shortlist keep their retriever rank (> K).
    """
    def z(x):
        x = x.float()
        return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)
    fused = (1 - weight) * z(retriever) + weight * z(other)
    t = fused.gather(1, target_pos.clamp(min=0)[:, None])
    r = (fused > t).sum(1) + 1 + ((fused == t).sum(1) - 1) / 2
    return torch.where(target_pos >= 0, r.float(), retriever_rank.float())


class CrossEncoderCHARM:
    """LLM + LoRA + per-dimension heads scoring (dialogue, candidate) pairs.

    ``heads=1`` scores relevance alone (training stage 1). ``heads=3`` adds the
    satisfaction and engagement heads and the dialogue gate that mixes them.
    """

    def __init__(self, model_path: str, device: str, lora_r: int = 16, lora_alpha: int = 32,
                 max_length: int = 256, dtype: torch.dtype = torch.float32, heads: int = 1,
                 profiles: Optional[Dict[str, str]] = None):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        local = os.path.isdir(model_path)
        self.tok = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
        # Left padding and truncation: the candidate sits at the end of every
        # sequence and the oldest turns are the ones dropped.
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=heads, dtype=dtype, local_files_only=local)
        base.config.pad_token_id = self.tok.pad_token_id
        cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_alpha,
                         lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg).to(device)
        self.heads = heads
        self.gate = None
        if heads > 1:
            # Starts favouring relevance (softmax of [2, 0, 0] ~ [0.78, 0.11, 0.11]);
            # zero weights, so every dialogue starts with the same mix. It reads the
            # unit-normalised dialogue state: the 7B's final hidden state has L1 norm
            # ~10^4, so on the raw state one AdamW step moves the gate by several logits.
            self.gate = torch.nn.Linear(base.config.hidden_size, heads).to(device)
            with torch.no_grad():
                self.gate.weight.zero_()
                self.gate.bias.copy_(torch.tensor([2.0] + [0.0] * (heads - 1)))
        # A bf16 backbone would leave the freshly initialised score head in bf16,
        # where AdamW steps of ~lr fall below its resolution. Keep the head in
        # float32; autocast handles the mixed-precision matmul.
        for p in self.head_parameters():
            p.data = p.data.float()
        self.device = device
        self.max_length = max_length
        self.profiles = dict(profiles or {})
        self.cand_budget = PROFILE_BUDGET if self.profiles else CAND_BUDGET
        self._ctx: Dict[Tuple[str, Optional[str]], List[int]] = {}
        self._cand: Dict[str, List[int]] = {}

    def score_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad and "score" in n]

    def gate_parameters(self):
        return list(self.gate.parameters()) if self.gate is not None else []

    def head_parameters(self):
        return self.score_parameters() + self.gate_parameters()

    def adapter_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad and "score" not in n]

    def train(self, mode: bool = True) -> None:
        self.model.train(mode)
        if self.gate is not None:
            self.gate.train(mode)

    def _prefix(self, dialogue: str, summary: Optional[str] = None) -> List[int]:
        key = (dialogue, summary)
        ctx = self._ctx.get(key)
        if ctx is None:
            tail = (self.tok(SUMMARY.format(summary=summary), add_special_tokens=False)
                    ["input_ids"][:SUMMARY_BUDGET] if summary else [])
            ids = self.tok(dialogue, add_special_tokens=False)["input_ids"]
            keep = max(self.max_length - self.cand_budget - len(tail), 1)
            ctx = ids[-keep:] + tail                   # keep the latest turns, then the summary
            self._ctx[key] = ctx
        return ctx

    def _suffix(self, title: str) -> List[int]:
        cand = self._cand.get(title)
        if cand is None:
            text = SUFFIX.format(title=title)
            if self.profiles.get(title):
                text += f" ({self.profiles[title]})"
            cand = self.tok(text, add_special_tokens=False)["input_ids"][:self.cand_budget]
            self._cand[title] = cand
        return cand

    def _autocast(self):
        if str(self.device).startswith("cuda"):
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.autocast("cpu", enabled=False)

    def grouped_scores(self, dialogues: Sequence[str], groups: Sequence[Sequence[str]],
                       summaries: Optional[Sequence[Optional[str]]] = None,
                       return_parts: bool = False):
        """``[B, G]`` scores for ``G`` candidates per dialogue, sharing the dialogue.

        Each dialogue is encoded once and its KV cache is shared by all its
        candidates, so a group costs one dialogue plus G short suffixes instead
        of G full sequences -- identical scores, ~G x less prefix compute, and
        differentiable (the cache tensors carry gradients back to the prefix).

        ``summaries`` (one per dialogue, or None) are appended to the dialogue.
        With ``return_parts`` also returns the per-dimension scores ``[B, G, H]``
        and the gate weights ``[B, H]``.
        """
        g = len(groups[0])
        assert all(len(x) == g for x in groups), "groups must have equal size"
        pad = self.tok.pad_token_id
        summaries = list(summaries) if summaries is not None else [None] * len(dialogues)

        prefixes = [self._prefix(d, sm) for d, sm in zip(dialogues, summaries)]
        width = max(len(p) for p in prefixes)
        p_ids = torch.full((len(prefixes), width), pad, dtype=torch.long)
        p_mask = torch.zeros((len(prefixes), width), dtype=torch.long)
        for i, p in enumerate(prefixes):          # left-pad: dialogue ends at the join
            p_ids[i, width - len(p):] = torch.tensor(p)
            p_mask[i, width - len(p):] = 1
        p_ids, p_mask = p_ids.to(self.device), p_mask.to(self.device)

        suffixes = [self._suffix(t) for grp in groups for t in grp]
        s_width = max(len(x) for x in suffixes)
        s_ids = torch.full((len(suffixes), s_width), pad, dtype=torch.long)
        s_mask = torch.zeros((len(suffixes), s_width), dtype=torch.long)
        for i, x in enumerate(suffixes):          # right-pad: pads follow the candidate
            s_ids[i, :len(x)] = torch.tensor(x)
            s_mask[i, :len(x)] = 1
        s_ids, s_mask = s_ids.to(self.device), s_mask.to(self.device)

        with self._autocast():
            out = self.model(input_ids=p_ids, attention_mask=p_mask, use_cache=True,
                             output_hidden_states=self.gate is not None)
            past = out.past_key_values
            # Left-padded, so the last position is every dialogue's final token.
            dialogue_state = out.hidden_states[-1][:, -1] if self.gate is not None else None
            past.batch_repeat_interleave(g)
            attn = torch.cat([p_mask.repeat_interleave(g, dim=0), s_mask], dim=1)
            # Sequence-classification pooling reads the last non-pad token of the
            # *suffix*, i.e. the end of each candidate.
            logits = self.model(input_ids=s_ids, attention_mask=attn,
                                past_key_values=past, use_cache=False).logits
        dims = logits.float().view(len(dialogues), g, self.heads)
        if self.gate is None:
            final, weights = dims[..., 0], torch.ones(len(dialogues), 1, device=dims.device)
        else:
            state = torch.nn.functional.normalize(dialogue_state.float(), dim=-1)
            weights = torch.softmax(self.gate(state), dim=-1)
            final = (dims * weights[:, None, :]).sum(-1)
        return (final, dims, weights) if return_parts else final

    @torch.no_grad()
    def score_groups(self, dialogues: Sequence[str], groups: Sequence[Sequence[str]],
                     dialogues_per_batch: int = 16,
                     summaries: Optional[Sequence[Optional[str]]] = None,
                     return_parts: bool = False):
        """Evaluation-mode :meth:`grouped_scores` over many dialogues."""
        was = self.model.training
        self.train(False)
        summaries = list(summaries) if summaries is not None else [None] * len(dialogues)
        try:
            out = [self.grouped_scores(dialogues[i:i + dialogues_per_batch],
                                       groups[i:i + dialogues_per_batch],
                                       summaries[i:i + dialogues_per_batch], return_parts=True)
                   for i in range(0, len(dialogues), dialogues_per_batch)]
        finally:
            self.train(was)
        if not out:
            empty = torch.empty(0, 0)
            return (empty, empty, empty) if return_parts else empty
        final, dims, weights = (torch.cat(x) for x in zip(*out))
        return (final, dims, weights) if return_parts else final

    def save(self, path: str, **config) -> None:
        """Adapter + score head (+ gate) and the settings needed to rebuild it."""
        import json

        self.model.save_pretrained(path)
        if self.gate is not None:
            torch.save(self.gate.state_dict(), os.path.join(path, "charm_gate.pt"))
        with open(os.path.join(path, "charm_config.json"), "w") as f:
            json.dump({"heads": self.heads, "max_length": self.max_length,
                       "profiles": bool(self.profiles), **config}, f, indent=2)

    def load_adapter(self, path: str) -> None:
        """Load a trained adapter + score head (+ gate) saved by :meth:`save`.

        A saved adapter with fewer heads than this model loads into the first
        rows of each head/gate tensor; the new heads start at zero, so the
        starting ranking and gate mix are exactly the saved adapter's (a copy
        of an existing row would make the new head's loss push that head's
        own ranking around through the shared adapter). Covers both the
        original 1 -> 3 warm start and 3 -> 4 (adding diversity).
        """
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        weights = load_file(os.path.join(path, "adapter_model.safetensors"))
        for k, v in list(weights.items()):
            if "score" in k and v.dim() >= 1 and v.size(0) != self.heads:
                if v.size(0) > self.heads:
                    raise ValueError(f"{k}: {v.size(0)} dimensions saved, model has {self.heads}")
                weights[k] = torch.cat([v, v.new_zeros(self.heads - v.size(0), *v.shape[1:])])
        result = set_peft_model_state_dict(self.model, weights)
        missing = [k for k in getattr(result, "unexpected_keys", []) or []]
        if missing:
            raise ValueError(f"adapter keys not in the model: {missing[:5]}")
        gate_file = os.path.join(path, "charm_gate.pt")
        if self.gate is not None and os.path.exists(gate_file):
            saved = torch.load(gate_file, map_location=self.device)
            n = saved["weight"].size(0)
            if n != self.heads:
                if n > self.heads:
                    raise ValueError(f"charm_gate.pt: {n} heads saved, model has {self.heads}")
                pad_h = self.heads - n
                saved = {
                    "weight": torch.cat([saved["weight"], saved["weight"].new_zeros(pad_h, saved["weight"].size(1))]),
                    "bias": torch.cat([saved["bias"], saved["bias"].new_zeros(pad_h)]),
                }
            self.gate.load_state_dict(saved)
        for p in self.head_parameters():
            p.data = p.data.float()

    def trainable_state(self) -> Dict[str, torch.Tensor]:
        state = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        if self.gate is not None:
            state.update({f"gate.{n}": p.detach().clone() for n, p in self.gate.named_parameters()})
        return state

    def load_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        params = dict(self.model.named_parameters())
        if self.gate is not None:
            params.update({f"gate.{n}": p for n, p in self.gate.named_parameters()})
        with torch.no_grad():
            for n, v in state.items():
                params[n].copy_(v)


def adapter_config(path: str) -> Dict:
    """Settings saved beside an adapter (``{}`` for stage-1 adapters saved without them)."""
    import json
    f = os.path.join(path, "charm_config.json")
    if not os.path.exists(f):
        return {}
    with open(f) as fh:
        return json.load(fh)
