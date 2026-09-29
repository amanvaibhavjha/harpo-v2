"""
Sequence pooling for HARPO.

Every downstream module (BRIDGE, CHARM, STAR, MAVEN, the ranking head) consumes a
fixed-size summary of the encoder's last hidden state. The original implementation
used a bare ``hidden_states[-1].mean(dim=1)``, which averages over padding as well
as content. With ``padding_side="left"`` and ``padding="max_length"`` that means a
short sequence is mostly padding, so the pooled vector is dominated by pad
embeddings and its scale varies with sequence length.

That is not cosmetic for preference learning: a chosen/rejected pair almost never
has the same length, so a reward model fed unmasked means can separate the two by
padding fraction alone, without reading either response.
"""

from typing import Optional

import torch


def masked_mean_pool(hidden_states: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean over the sequence axis, ignoring padded positions.

    Args:
        hidden_states: ``[batch, seq, hidden]``, or ``[batch, hidden]`` (returned
            unchanged, so callers can stay agnostic about what they were handed).
        attention_mask: ``[batch, seq]`` with 1 for real tokens. When ``None`` this
            degrades to an unmasked mean -- only correct if nothing is padded.

    Returns:
        ``[batch, hidden]``
    """
    if hidden_states.dim() == 2:
        return hidden_states
    if hidden_states.dim() != 3:
        raise ValueError(f"expected [batch, seq, hidden], got {tuple(hidden_states.shape)}")

    if attention_mask is None:
        return hidden_states.mean(dim=1)

    if attention_mask.dim() != 2 or attention_mask.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}"
        )

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    # An all-padding row would divide by zero; clamp so it yields zeros not NaN.
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
