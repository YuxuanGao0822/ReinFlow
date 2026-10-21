"""ReinFlow-compatible online PPO for stochastic data-space refinement.

The PPO buffer has K+1 slots:
  chain[:, 0]     = fixed source Gaussian z,
  chain[:, 1:K+1] = raw, *unclipped* refinement proposals.
At stage zero the previous proposal is zero. At stage r>0 it is
chain[:, r]. This reuses PPOFlowBuffer without lossy recomputation of z.
The executed action is clipped only after sampling the raw latent chain.
"""
import copy
import logging
import torch
from torch import nn
from torch.distributions import Normal
from model.flow.ft_ppo.ppoflow import PPOFlow

log = logging.getLogger(__name__)


class NoisyDataSpace(nn.Module):
    """Generator plus bounded stage-and-action-dimension exploration std."""
    def __init__(self, policy, steps, min_std, max_std):
        super().__init__()
        if min_std <= 0 or max_std < min_std:
            raise ValueError("invalid stochastic refinement std bounds")
        self.policy = policy
        self.denoising_steps = steps
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.std_logits = nn.Parameter(torch.zeros(
            steps, policy.horizon_steps, policy.action_dim))

    def get_std(self, stage, previous):
        b = previous.shape[0]
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(
            self.std_logits[stage])
        return std.unsqueeze(0).expand(b, -1, -1)


class PPODataSpace(PPOFlow):
    """Drop-in PPOFlow replacement with non-ODE refinement transitions."""

    def load_policy(self, network_path, use_ema=True):
        if not network_path:
            raise ValueError("Data-Space PPO needs a pretrained DMD checkpoint")
        payload = torch.load(network_path, map_location="cpu", weights_only=True)
        weights = payload["ema" if use_ema and "ema" in payload else "model"]
        prefix = "generator."
        generator_weights = {k[len(prefix):]: v for k, v in weights.items()
                             if k.startswith(prefix)}
        if not generator_weights:
            if "velocity.time_embedding.1.weight" in weights:
                generator_weights = weights
            else:
                raise ValueError("Checkpoint contains no generator.* weights")
        self.actor_old.load_state_dict(generator_weights, strict=True)

    def init_actor_ft(self, policy_copy):
        if policy_copy.refinement_steps < self.inference_steps:
            raise ValueError("DMD checkpoint trained for fewer refinement steps")
        self.actor_ft = NoisyDataSpace(
            policy_copy, self.inference_steps,
            self.min_logprob_denoising_std,
            self.max_logprob_denoising_std).to(self.device)

    def _joint_logprob(self, z, proposals, cond, get_entropy=False,
                       account_for_initial_stochasticity=True,
                       normalize_denoising_horizon=False,
                       normalize_act_space_dimension=False):
        # Recomputes exact Gaussian density for the stored RAW latent chain.
        b, k = proposals.shape[:2]
        if k != self.inference_steps:
            raise ValueError("refinement chain length mismatch")
        total = z.new_zeros(b)
        entropy = z.new_zeros(b)
        n = 0
        if account_for_initial_stochasticity:
            normal0 = Normal(torch.zeros_like(z), torch.ones_like(z))
            total = total + normal0.log_prob(z).flatten(1).sum(1)
            if get_entropy:
                entropy = entropy + normal0.entropy().flatten(1).sum(1)
            n += 1
        std_means = []
        for stage in range(k):
            prev = (torch.zeros_like(z) if stage == 0
                    else proposals[:, stage - 1])
            mean = self.actor_ft.policy(prev, z, cond, stage)
            std = self.actor_ft.get_std(stage, prev)
            dist = Normal(mean, std)
            total = total + dist.log_prob(proposals[:, stage]).flatten(1).sum(1)
            if get_entropy:
                entropy = entropy + dist.entropy().flatten(1).sum(1)
            std_means.append(std.mean())
            n += 1
        if normalize_denoising_horizon:
            total = total / n
            if get_entropy:
                entropy = entropy / n
        if normalize_act_space_dimension:
            total = total / self.act_dim_total
            if get_entropy:
                entropy = entropy / self.act_dim_total
        if get_entropy:
            return total, entropy, torch.stack(std_means).mean()
        return total, torch.stack(std_means).mean()

    @torch.no_grad()
    def get_actions(self, cond, eval_mode, save_chains=False,
                    normalize_denoising_horizon=False,
                    normalize_act_space_dimension=False,
                    clip_intermediate_actions=True,
                    account_for_initial_stochasticity=True, ret_logprob=True):
        b = cond["state"].shape[0]
        z = torch.randn(b, self.horizon_steps, self.action_dim, device=self.device)
        prev = torch.zeros_like(z)
        chain = [z]
        for stage in range(self.inference_steps):
            mean = self.actor_ft.policy(prev, z, cond, stage)
            std = self.actor_ft.get_std(stage, prev)
            # No clipping of internal states: their Gaussian density must
            # match the data actually placed in the replay buffer.
            prev = mean if eval_mode else mean + torch.randn_like(mean) * std
            chain.append(prev)
        raw = torch.stack(chain, dim=1)
        # Only the environment-facing action is clipped.
        action = prev.clamp(self.act_min, self.act_max)
        if ret_logprob:
            lp, _ = self._joint_logprob(
                z, raw[:, 1:], cond,
                account_for_initial_stochasticity=account_for_initial_stochasticity,
                normalize_denoising_horizon=normalize_denoising_horizon,
                normalize_act_space_dimension=normalize_act_space_dimension)
            return (action, raw, lp) if save_chains else (action, lp)
        return (action, raw) if save_chains else action

    def get_logprobs(self, cond, x_chain, get_entropy=False,
                     normalize_denoising_horizon=False,
                     normalize_act_space_dimension=False,
                     clip_intermediate_actions=True,
                     verbose_entropy_stats=True,
                     account_for_initial_stochasticity=True,
                     get_chains_stds=True, **kwargs):
        if x_chain.ndim != 4 or x_chain.shape[1] != self.inference_steps + 1:
            raise ValueError("expected chain [B, K+1, horizon, action_dim]")
        result = self._joint_logprob(
            x_chain[:, 0], x_chain[:, 1:], cond,
            get_entropy=get_entropy,
            account_for_initial_stochasticity=account_for_initial_stochasticity,
            normalize_denoising_horizon=normalize_denoising_horizon,
            normalize_act_space_dimension=normalize_act_space_dimension)
        if get_entropy:
            prob, entropy, mean_std = result
            return (prob, entropy, mean_std) if get_chains_stds else (prob, entropy)
        prob, mean_std = result
        return (prob, mean_std) if get_chains_stds else prob
