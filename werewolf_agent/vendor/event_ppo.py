"""MAPPO-derived PPO / value / GAE primitives (MIT), decentralized requests.

Copyright (c) 2021 MAPPO contributors. See MAPPO_LICENSE and SOURCES.md.
No centralized observations, synchronous team buffer or death-as-done masks.
"""

import torch


def masked_distribution(logits, legal):
    if (legal.dtype != torch.bool or legal.shape != logits.shape
            or not bool(legal.any(dim=-1).all()) or not bool(torch.isfinite(logits[legal]).all())):
        raise ValueError("Invalid candidate logits/mask")
    return torch.distributions.Categorical(logits=logits.masked_fill(~legal, float("-inf")))


def clipped_policy_loss(new_logp, old_logp, advantages, clip=.2):
    ratio = torch.exp(new_logp - old_logp)
    return -torch.minimum(ratio * advantages,
                          torch.clamp(ratio, 1. - clip, 1. + clip) * advantages)


def clipped_value_loss(values, old_values, targets, clip=.2):
    clipped = old_values + torch.clamp(values - old_values, -clip, clip)
    return .5 * torch.maximum((targets - values).square(), (targets - clipped).square())


def event_gae(rewards, values, next_values, bootstrap_mask, trace_mask, gamma=1., lam=.95):
    """One owner-trajectory fragment, ordered requests; masks have distinct roles.

    Ordinary=(1,1), clean truncation=(1,0), true faction terminal=(0,0).
    The terminal outcome is attached to a dead seat's final actionable request.
    """
    arrays = (rewards, values, next_values, bootstrap_mask, trace_mask)
    if (any(item.ndim != 1 or item.shape != rewards.shape for item in arrays)
            or any(not bool(torch.isfinite(item).all()) for item in arrays)
            or any(not bool(((item == 0) | (item == 1)).all())
                   for item in (bootstrap_mask, trace_mask))
            or bool((trace_mask > bootstrap_mask).any())
            or not 0 <= gamma <= 1 or not 0 <= lam <= 1):
        raise ValueError("Invalid event-time trajectory")
    advantages = torch.zeros_like(values)
    carry = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * bootstrap_mask[index] * next_values[index] - values[index]
        carry = delta + gamma * lam * trace_mask[index] * carry
        advantages[index] = carry
    return advantages.detach(), (advantages + values).detach()


def exact_categorical_kl(old_probs, new_probs, legal):
    if (old_probs.ndim < 1 or legal.dtype != torch.bool
            or old_probs.shape != new_probs.shape or legal.shape != old_probs.shape
            or not bool(legal.any(dim=-1).all())
            or not bool(torch.isfinite(old_probs).all())
            or not bool(torch.isfinite(new_probs).all())
            or bool((old_probs < 0).any()) or bool((new_probs < 0).any())
            or not bool((old_probs[~legal] == 0).all())
            or not bool((new_probs[~legal] == 0).all())
            or not bool(torch.isclose(old_probs.sum(dim=-1), torch.ones_like(old_probs.sum(dim=-1)),
                                      atol=1e-6, rtol=1e-6).all())
            or not bool(torch.isclose(new_probs.sum(dim=-1), torch.ones_like(new_probs.sum(dim=-1)),
                                      atol=1e-6, rtol=1e-6).all())):
        raise ValueError("KL requires valid normalized categorical distributions")
    positive = legal & (old_probs > 0)
    log_ratio = torch.zeros_like(old_probs)
    log_ratio[positive] = torch.log(old_probs[positive]) - torch.log(new_probs[positive])
    return (old_probs * log_ratio).sum(dim=-1)
