from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Mean pair loss from PDF pp. 3–4; inputs are response-token sums."""

    policy_margin = (
        policy_chosen_logp
        - policy_rejected_logp
    )

    # Reference scores are constants, even if a caller accidentally supplies a graph.
    ref_margin = ref_chosen_logp.detach() - ref_rejected_logp.detach()

    # log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected).
    # Adding the reference margin would reward its likelihood difference twice.
    preference_margin = (policy_chosen_logp - ref_chosen_logp.detach()) - (
        policy_rejected_logp - ref_rejected_logp.detach()
    )
    logits = beta * preference_margin

    loss = -F.logsigmoid(
        logits
    ).mean()

    return loss, {
        "logit_mean":
            logits.detach().mean(),

        "policy_margin_mean":
            policy_margin.detach().mean(),

        "preference_margin_mean": preference_margin.detach().mean(),
        "ref_margin_mean": ref_margin.mean(),

        "preference_accuracy": (
            preference_margin > 0
        ).float().mean().detach(),
    }
