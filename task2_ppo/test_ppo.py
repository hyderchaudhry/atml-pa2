"""Offline CPU smoke tests: no pretrained models, generation, or course experiments."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from common.data import load_yaml
from task2_ppo import continue_train as train
from task2_ppo import evaluate as evaluation
from task2_ppo import analyze_clipping as clipping
from task2_ppo.analyze_clipping import cached_tensors, clipping_statistics
from task2_ppo.ppo import (
    compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss,
)


class ObjectiveTests(unittest.TestCase):
    def test_clipping_signs_gradients_and_padding(self):
        # All four sign/tail combinations, an in-range token, and ignored padding.
        new = torch.tensor([[1.5, 0.5, 1.5, 0.5, 1.0, 10.0]]).log().requires_grad_()
        old = torch.zeros_like(new)
        advantage = torch.tensor([[2., 2., -2., -2., 1., 999.]])
        mask = torch.tensor([[1., 1., 1., 1., 1., 0.]])
        loss, ratio, fraction = ppo_policy_loss(new, old, advantage, mask, eps=0.2)
        # min surrogates: 2.4, 1.0, -3.0, -1.6, 1.0 => loss = 0.04.
        self.assertAlmostEqual(loss.item(), 0.04, places=6)
        self.assertAlmostEqual(fraction.item(), 0.8, places=6)
        self.assertFalse(ratio.requires_grad)
        loss.backward()
        torch.testing.assert_close(new.grad, torch.tensor([[0., -0.2, 0.6, 0., -0.2, 0.]]))
        stats = clipping_statistics(new.detach(), {"old": old, "advantages": advantage, "mask": mask}, 0.2)
        self.assertAlmostEqual(stats["affected_token_fraction"], 0.8, places=6)
        self.assertAlmostEqual(stats["surrogate_constrained_fraction"], 0.4, places=6)
        for epsilon in (0.05, 0.2, 0.5):
            _, _, fraction = ppo_policy_loss(old, old, advantage, mask, eps=epsilon)
            self.assertEqual(fraction.item(), 0.)

    def test_terminal_reward_gae_and_masked_value_loss(self):
        mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
        old = torch.ones_like(mask)
        rewards = shaped_rewards(torch.tensor([3., 6.]), old, torch.zeros_like(old), mask, 0.1)
        torch.testing.assert_close(rewards, torch.tensor([[-0.1, 2.9, 0.], [-0.1, -0.1, 5.9]]))
        values = torch.tensor([[0.5, 1., 999.], [1., 2., 3.]])
        advantage, returns = compute_gae(rewards, values, mask, gamma=1., lam=1.)
        torch.testing.assert_close(advantage, torch.tensor([[2.3, 1.9, 0.], [4.7, 3.8, 2.9]]))
        torch.testing.assert_close(returns * mask, torch.tensor([[2.8, 2.9, 0.], [5.7, 5.8, 5.9]]))
        normalized = normalize_advantages(advantage, mask)
        self.assertAlmostEqual(normalized[mask.bool()].mean().item(), 0., places=6)
        self.assertAlmostEqual(normalized[mask.bool()].std(unbiased=False).item(), 1., places=6)
        pred = (returns * mask).requires_grad_()
        value_mse_loss(pred, returns, mask).backward()
        self.assertEqual(pred.grad[0, 2].item(), 0.)
        self.assertTrue(torch.isfinite(normalize_advantages(torch.ones(1, 1), torch.ones(1, 1))).all())

    def test_cached_padding_and_advantages(self):
        rows = [
            {"response_tokens": n, "old_logprobs": torch.zeros(n), "ref_logprobs": torch.zeros(n),
             "values": torch.zeros(n), "effective_terminal_reward": float(n)} for n in (1, 3)
        ]
        batch = cached_tensors(rows, {"kl_beta": 0.1, "gamma": 1., "gae_lambda": 1.})
        torch.testing.assert_close(batch["mask"], torch.tensor([[1., 0., 0.], [1., 1., 1.]]))
        torch.testing.assert_close(batch["returns"], torch.tensor([[1., 0., 0.], [3., 3., 3.]]))
        rows[0]["old_logprobs"] = torch.zeros(2)
        with self.assertRaises(ValueError):
            cached_tensors(rows, {"kl_beta": 0.1, "gamma": 1., "gae_lambda": 1.})

    def test_critic_prefix_alignment_and_lora_head_gradients(self):
        # Tiny randomly initialized library models test the actual released value helper.
        from peft import get_peft_model
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        from common.models import make_value_lora_config, token_values, value_parameter_groups
        cfg = load_yaml("configs/ppo.yaml")
        critic = get_peft_model(Qwen2ForSequenceClassification(Qwen2Config(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=0,
        )), make_value_lora_config(cfg)).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        rollout = {"sequences": ids, "attention_mask": torch.ones_like(ids),
                   "prompt_width": 2, "response_ids": ids[:, 2:]}
        actual = train.rollout_values(critic, rollout)
        expected = token_values(critic, ids, torch.ones_like(ids))[:, 1:3]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        groups = value_parameter_groups(critic, 1e-4, 3e-4)
        self.assertEqual([g["name"] for g in groups], ["critic_lora", "critic_head"])
        self.assertTrue(all(p.grad is not None for g in groups for p in g["params"]))


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.embedding(input_ids))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "fake_adapter.json").write_text('{}')


def fake_generation(*args, **kwargs):
    # Padded response tests mask handling; inference tensors test safe conversion for autograd.
    with torch.inference_mode():
        sequences = torch.tensor([[1, 2, 3, 4, 0]])
    return {"sequences": sequences, "attention_mask": torch.ones_like(sequences), "prompt_width": 2,
            "response_ids": sequences[:, 2:], "response_mask": torch.tensor([[1., 1., 0.]]),
            "responses": ["synthetic"], "terminated_with_eos": [True], "truncated": [False],
            "response_lengths": [2]}


class PlumbingTests(unittest.TestCase):
    def test_fp16_critic_head_can_be_unscaled(self):
        from peft import get_peft_model
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        from common.models import make_value_lora_config
        cfg = load_yaml("configs/ppo.yaml")
        critic = get_peft_model(Qwen2ForSequenceClassification(Qwen2Config(
            vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, num_labels=1, pad_token_id=0,
        )).half(), make_value_lora_config(cfg)).eval()
        with patch.object(train, "load_tokenizer"), \
             patch.object(train, "load_policy", return_value=TinyPolicy()), \
             patch.object(train, "load_value_model", return_value=critic), \
             patch.object(train, "load_reward_model", return_value=(None, None)), \
             patch.object(train, "read_jsonl", return_value=[]):
            bundle = train.prepare_ppo_continuation("configs/ppo.yaml")
        self.assertTrue(all(p.dtype == torch.float32 for p in critic.parameters() if p.requires_grad))
        ids = torch.tensor([[1, 2, 3, 4]])
        rollout = {"sequences": ids, "attention_mask": torch.ones_like(ids),
                   "prompt_width": 2, "response_ids": ids[:, 2:]}
        values = train.rollout_values(critic, rollout)
        self.assertTrue(torch.isfinite(values).all())
        scaler = torch.amp.GradScaler("cpu", init_scale=1.)
        scaler.scale(values.square().mean()).backward()
        scaler.unscale_(bundle["value_optimizer"])
        scaler.step(bundle["value_optimizer"])
        scaler.update()

    def test_release_configuration_and_checkpoint_restoration(self):
        cfg = load_yaml("configs/ppo.yaml")
        self.assertEqual((cfg["updates"], cfg["fork_updates"], cfg["seed"]), (20, 8, 6304))
        self.assertEqual(cfg["clip_values"], [0.05, 0.20, 0.50])
        self.assertEqual(cfg["kl_values"], [0., 0.10, 0.20])
        critic = torch.nn.Module()
        critic.score = torch.nn.Linear(1, 1)
        with patch.object(train, "load_tokenizer"), \
             patch.object(train, "load_policy", return_value=TinyPolicy()) as policy_loader, \
             patch.object(train, "load_value_model", return_value=critic) as value_loader, \
             patch.object(train, "load_reward_model", return_value=(None, None)), \
             patch.object(train, "read_jsonl", return_value=[]):
            bundle = train.prepare_ppo_continuation("configs/ppo.yaml")
        self.assertEqual(policy_loader.call_args.kwargs, {
            "adapter_path": "checkpoints/ppo_midpoint_policy", "trainable": True,
        })
        self.assertEqual(value_loader.call_args.args[1], "checkpoints/ppo_midpoint_value")
        self.assertEqual(value_loader.call_args.kwargs["train_mode"], "lora_head")
        self.assertEqual(bundle["policy_optimizer"].param_groups[0]["lr"], 3e-6)
        self.assertEqual(bundle["value_optimizer"].param_groups[0]["lr"], 3e-4)

    def test_mocked_continuation_and_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="ppo-smoke-") as tmp:
            cfg = load_yaml("configs/ppo.yaml")
            cfg.update(output=f"{tmp}/adapter", results_dir=f"{tmp}/results")
            rows = [{"prompt_id": f"p{i}", "source_index": i, "messages": [{"role": "user", "content": str(i)}]}
                    for i in range(2)]
            policy, critic = TinyPolicy(), torch.nn.Linear(1, 1)
            tokenizer = Mock()
            bundle = {"policy": policy, "value_model": critic, "tokenizer": tokenizer,
                      "reward_model": None, "reward_tokenizer": None, "prompt_rows": rows,
                      "policy_optimizer": torch.optim.AdamW(policy.parameters(), lr=0.001),
                      "value_optimizer": torch.optim.AdamW(critic.parameters(), lr=0.001)}
            weights_before = policy.embedding.weight.detach().clone()
            def values(model, rollout):
                return model(torch.ones(1, 3, 1)).squeeze(-1)
            with patch.object(train, "load_yaml", side_effect=lambda _: copy.deepcopy(cfg)), \
                 patch.object(train, "prepare_ppo_continuation", return_value=bundle) as prepare, \
                 patch.object(train, "batch_generate", side_effect=fake_generation) as generate, \
                 patch.object(train, "score_reward_pairs", return_value=torch.tensor([2.])), \
                 patch.object(train, "rollout_values", side_effect=values):
                summary = train.run_ppo("unused", updates=2, clip_epsilon=0.05, kl_beta=0., run_name="smoke")
                prepare.assert_called_once_with("unused")
                self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 512)
                with self.assertRaises(FileExistsError):
                    train.run_ppo("unused", updates=2, run_name="smoke")
            self.assertFalse(torch.equal(weights_before, policy.embedding.weight))
            self.assertEqual(summary["completed_updates"], 2)
            self.assertEqual(summary["epsilon"], 0.05)
            self.assertEqual(summary["beta_kl"], 0.)
            self.assertIsNone(summary["peak_allocated_bytes"])
            self.assertIsNone(summary["peak_reserved_bytes"])
            self.assertGreater(summary["wall_clock_seconds"], 0.)
            result = Path(tmp) / "results/smoke"
            trajectory = [json.loads(line) for line in (result / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual([r["prompt_ids"][0]["prompt_id"] for r in trajectory], ["p0", "p1"])
            for record in trajectory:
                for key in ("learned_reward", "kl", "policy_loss", "value_loss", "entropy", "gradient_norm", "clip_fraction", "response_length"):
                    self.assertIn(key, record)
                self.assertEqual(record["response_length"], 2.)
                self.assertEqual(record["epochs"][0]["clip_fraction"], 0.)
            eval_bundle = {"cfg": cfg, "rows": rows, "tokenizer": tokenizer,
                           "policy": policy, "reward": (None, None)}
            def uneven_generation(*args, **kwargs):
                rollout = fake_generation()
                if args[2][0][0]["content"] == "1":
                    rollout["response_mask"] = torch.ones(1, 3)
                    rollout["response_lengths"] = [3]
                return rollout
            with patch.object(evaluation, "load_yaml", return_value=cfg), \
                 patch.object(evaluation, "load_evaluation_bundle", return_value=eval_bundle), \
                 patch.object(train, "batch_generate", side_effect=uneven_generation) as generate, \
                 patch.object(evaluation, "score_reward_pairs", return_value=torch.tensor([2.])):
                metrics = evaluation.evaluate("unused", cfg["output"], "smoke")
                self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 768)
            self.assertEqual(metrics["learned_reward"], 2.)
            self.assertEqual(metrics["epsilon"], 0.05)
            self.assertEqual(metrics["beta_kl"], 0.)
            examples = [json.loads(line) for line in (result / "generations.jsonl").read_text().splitlines()]
            self.assertEqual(examples[0]["response_token_ids"], [3, 4])
            self.assertEqual(len(examples), 2)
            self.assertEqual(metrics["response_length"], 2.5)
            self.assertEqual(metrics["response_length_std"], 0.5)
            self.assertAlmostEqual(metrics["entropy"],
                                   (2 * examples[0]["entropy"] + 3 * examples[1]["entropy"]) / 5)
            self.assertTrue((Path(tmp) / "adapter/fake_adapter.json").exists())

    def test_cached_analysis_saves_all_epsilons_without_generation(self):
        with tempfile.TemporaryDirectory(prefix="ppo-cache-smoke-") as tmp:
            cfg = load_yaml("configs/ppo.yaml")
            cfg["results_dir"] = tmp
            rows = [{"prompt_id": "p", "source_index": 1, "response_tokens": 2,
                     "old_logprobs": torch.zeros(2), "ref_logprobs": torch.zeros(2),
                     "values": torch.zeros(2), "effective_terminal_reward": 2.}]
            with patch.object(clipping, "load_yaml", return_value=cfg), \
                 patch.object(clipping, "load_cached_rollouts", return_value=rows), \
                 patch.object(clipping, "read_jsonl", return_value=rows), \
                 patch.object(clipping, "load_tokenizer"), \
                 patch.object(clipping, "load_policy", return_value=TinyPolicy()), \
                 patch.object(clipping, "reconstruct_cached_response", return_value={}), \
                 patch.object(clipping, "rollout_logprobs", return_value=torch.tensor([[1.3, 0.9]]).log()), \
                 patch.object(train, "batch_generate", side_effect=AssertionError("Must reuse cache")):
                clipping.analyze_cached("unused")
            result = json.loads((Path(tmp) / "clipping/cached_rollout.json").read_text())
            self.assertEqual([r["epsilon"] for r in result["conditions"]], [0.05, 0.2, 0.5])
            self.assertEqual(result["candidate_adapter"], cfg["output"])
            self.assertEqual(result["valid_response_tokens"], 2)

    def test_fork_commands_only_change_requested_parameter(self):
        cfg = load_yaml("configs/ppo.yaml")
        for flag, values, study in (("--clip-epsilon", cfg["clip_values"], "clipping"),
                                    ("--kl-beta", cfg["kl_values"], "kl")):
            with patch.object(train.subprocess, "run") as launch, \
                 patch.object(train, "load_json", return_value={"epsilon": 0.2, "beta_kl": 0.1,
                                                              "mean_gradient_norm": 1., "max_gradient_norm": 2.,
                                                              "skipped_policy_steps": 0, "skipped_value_steps": 0}), \
                 patch.object(train, "save_json") as save:
                train.run_forks("configs/ppo.yaml", flag, values, study)
            self.assertEqual(launch.call_count, 6)
            for i, value in enumerate(values):
                command = launch.call_args_list[2 * i].args[0]
                self.assertEqual(command[command.index(flag) + 1], str(value))
                self.assertEqual(command[command.index("--updates") + 1], "8")
                other = "--kl-beta" if flag == "--clip-epsilon" else "--clip-epsilon"
                self.assertNotIn(other, command)
                self.assertEqual(launch.call_args_list[2 * i + 1].args[0][2], "task2_ppo.evaluate")
            self.assertEqual(len(save.call_args.args[1]["conditions"]), 3)


if __name__ == "__main__":
    unittest.main()
