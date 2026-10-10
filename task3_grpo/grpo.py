from __future__ import annotations

import torch

from common.metrics import masked_mean, sample_entropy


ADVANTAGE_EPS = 1e-6


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = ADVANTAGE_EPS):
    """Return one scalar advantage per sampled completion.

    `group_ids[i]` identifies which prompt produced reward `rewards[i]`.
    Use population standard deviation and the manual's additive epsilon.
    """
    if rewards.ndim != 1 or rewards.shape != group_ids.shape or eps <= 0:
        raise ValueError("Expected aligned reward/group vectors and positive epsilon")
    rewards = rewards.float()
    group_ids = group_ids.to(rewards.device)
    advantages = torch.zeros_like(rewards)
    for group_id in torch.unique(group_ids):
        members = group_ids == group_id
        group = rewards[members]
        advantages[members] = (group - group.mean()) / (group.std(unbiased=False) + eps)
    return advantages


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None or max_completion_length <= 0:
            raise ValueError("dr_grpo requires positive max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    if truncated.shape != (token_mask.shape[0],):
        raise ValueError("Expected one truncation flag per completion")
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep


def normalization_statistics(new_logp, old_logp, advantages, mask, eps, loss_type, cap):
    """Exact policy-surrogate derivative w.r.t. sampled-token log probabilities.

    This measures normalization's gradient allocation, not a parameter-gradient norm.
    It excludes the unchanged KL term and includes the completion-batch averaging.
    """
    logp = new_logp.detach().requires_grad_(True)
    loss, _ = grpo_policy_loss(logp, old_logp.detach(), advantages.detach(), mask,
                               logp.detach(), eps, 0.0, loss_type, cap)
    gradient, = torch.autograd.grad(loss, logp)
    lengths = mask.sum(-1)
    return {
        "surrogate_logp_gradient_l1": gradient.abs().sum(-1).tolist(),
        "surrogate_logp_gradient_l2": gradient.square().sum(-1).sqrt().tolist(),
        "mean_absolute_token_gradient": (gradient.abs().sum(-1) / lengths.clamp_min(1)).tolist(),
    }
