"""Slurm-only deterministic checks of the attributed attention / PPO core."""

import unittest

import torch

from werewolf_agent.vendor.event_ppo import (
    clipped_policy_loss, clipped_value_loss, event_gae, exact_categorical_kl,
    masked_distribution,
)
from werewolf_agent.vendor.slot_attention import SlotBlock


class PrimitiveChecks(unittest.TestCase):
    def test_padding_does_not_change_real_slot_outputs(self):
        torch.manual_seed(17)
        block = SlotBlock().eval()
        real = torch.randn(1, 3, 128)
        baseline = block(real, torch.ones(1, 3, dtype=torch.bool))
        padded = torch.cat((real, torch.randn(1, 7, 128) * 100), dim=1)
        mask = torch.tensor([[True] * 3 + [False] * 7])
        result = block(padded, mask)
        torch.testing.assert_close(result[:, :3], baseline, atol=1e-6, rtol=1e-5)
        self.assertTrue(bool((result[:, 3:] == 0).all()))
        padded[:, 3:] = float("nan")
        torch.testing.assert_close(block(padded, mask)[:, :3], baseline, atol=1e-6, rtol=1e-5)

    def test_slot_permutation_equivariance(self):
        torch.manual_seed(19)
        block = SlotBlock().eval()
        inputs = torch.randn(2, 5, 128)
        mask = torch.ones(2, 5, dtype=torch.bool)
        order = torch.tensor([4, 1, 0, 3, 2])
        expected = block(inputs, mask)[:, order]
        actual = block(inputs[:, order], mask[:, order])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_singleton_has_zero_actor_gradient_and_padding_zero_probability(self):
        logits = torch.tensor([[.3, 4., -5.]], requires_grad=True)
        distribution = masked_distribution(logits, torch.tensor([[True, False, False]]))
        loss = clipped_policy_loss(distribution.log_prob(torch.tensor([0])),
                                   torch.zeros(1), torch.ones(1)).sum()
        loss.backward()
        torch.testing.assert_close(distribution.probs, torch.tensor([[1., 0., 0.]]))
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_faction_outcome_survives_death_and_learned_values_affect_gae(self):
        reward = torch.tensor([0., 1.])
        boot, trace = torch.tensor([1., 0.]), torch.tensor([1., 0.])
        zero_adv, zero_target = event_gae(reward, torch.zeros(2), torch.zeros(2), boot, trace)
        adv, target = event_gae(reward, torch.tensor([.2, .4]), torch.tensor([.4, 0.]), boot, trace)
        # Last request may precede death; its eventual faction win is still +1.
        self.assertAlmostEqual(float(target[-1]), 1.)
        self.assertGreater(float(target[0]), 0.)
        self.assertFalse(torch.allclose(adv, zero_adv))
        self.assertFalse(adv.requires_grad)

    def test_clean_fragment_bootstraps_without_tracing_into_next_fragment(self):
        adv, targets = event_gae(torch.tensor([0., -1.]), torch.tensor([.1, .2]),
                                 torch.tensor([.6, 0.]), torch.tensor([1., 0.]),
                                 torch.tensor([0., 0.]))
        torch.testing.assert_close(adv, torch.tensor([.5, -1.2]))
        torch.testing.assert_close(targets, torch.tensor([.6, -1.]))

    def test_value_clipping_and_exact_full_slate_kl(self):
        loss = clipped_value_loss(torch.tensor([1.]), torch.tensor([0.]), torch.tensor([1.]))
        torch.testing.assert_close(loss, torch.tensor([.32]))
        old, new = torch.tensor([[.25, .75, 0.]]), torch.tensor([[.5, .5, 0.]])
        kl = exact_categorical_kl(old, new, torch.tensor([[True, True, False]]))
        expected = .25 * torch.log(torch.tensor(.5)) + .75 * torch.log(torch.tensor(1.5))
        torch.testing.assert_close(kl, expected.reshape(1))

    def test_invalid_mask_and_fake_terminal_trace_reject(self):
        with self.assertRaises(ValueError):
            masked_distribution(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            event_gae(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                      torch.zeros(1), torch.ones(1))

    def test_kl_rejects_invalid_mass_but_valid_support_loss_is_infinite(self):
        good = torch.tensor([[.5, .5]])
        legal = torch.ones(1, 2, dtype=torch.bool)
        for bad in (torch.tensor([[-.5, 1.5]]), torch.tensor([[.2, .2]]),
                    torch.tensor([[float("nan"), .5]]), torch.tensor([[float("inf"), .5]])):
            with self.assertRaises(ValueError):
                exact_categorical_kl(good, bad, legal)
            with self.assertRaises(ValueError):
                exact_categorical_kl(bad, good, legal)
        with self.assertRaises(ValueError):
            exact_categorical_kl(good, good, legal.float())
        self.assertTrue(bool(torch.isinf(exact_categorical_kl(good, torch.tensor([[1., 0.]]), legal)).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
