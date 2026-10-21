"""Conditional data-space DMD with frozen real flow and trainable fake flow."""
import copy
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import nn
from .iteration import DataSpaceGenerator


class DataSpaceDMD(nn.Module):
    def __init__(self, network: nn.Module, device: str, horizon_steps: int,
                 action_dim: int, refinement_steps: int = 4, fake_updates: int = 4,
                 time_alpha: float = 2.5, time_beta: float = 1.,
                 min_time: float = .05, max_time: float = .95,
                 obs_dim=None, act_min=-1., act_max=1.,
                 max_denoising_steps=4, seed=42, sample_t_type="uniform"):
        super().__init__()
        self.generator = DataSpaceGenerator(copy.deepcopy(network),
                                            horizon_steps, action_dim,
                                            refinement_steps)
        self.real_flow = copy.deepcopy(network)
        self.fake_flow = copy.deepcopy(network)
        self.fake_stage = nn.Embedding(refinement_steps, horizon_steps * action_dim)
        nn.init.zeros_(self.fake_stage.weight)
        self.real_flow.requires_grad_(False)
        self.real_flow.eval()
        self.refinement_steps = int(refinement_steps)
        self.fake_updates = int(fake_updates)
        self.time_alpha, self.time_beta = float(time_alpha), float(time_beta)
        self.min_time, self.max_time = min_time, max_time
        self.horizon_steps, self.action_dim = horizon_steps, action_dim
        self.act_range = (act_min, act_max)

    def train(self, mode=True):
        super().train(mode)
        self.real_flow.eval()
        return self

    def initialize_from_reflow(self, checkpoint, use_ema=True):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source = payload["ema" if use_ema and "ema" in payload else "model"]
        weights = {k[len("network."):]: v for k, v in source.items()
                   if k.startswith("network.")}
        if not weights:
            raise ValueError("Expected ReFlow checkpoint state_dict keys 'network.*'")
        self.real_flow.load_state_dict(weights, strict=True)
        self.fake_flow.load_state_dict(weights, strict=True)
        self.generator.velocity.load_state_dict(weights, strict=True)
        self.real_flow.requires_grad_(False).eval()

    def sample_time(self, b, device):
        t = torch.distributions.Beta(self.time_alpha, self.time_beta).sample((b,))
        return t.to(device).clamp(self.min_time, self.max_time)

    def fake_velocity(self, xt, t, cond, stage):
        b = xt.shape[0]
        ids = torch.full((b,), stage, dtype=torch.long, device=xt.device)
        offset = self.fake_stage(ids).reshape_as(xt)
        return self.fake_flow(xt + offset, t, cond)

    def fake_loss(self, action_shape_reference, cond, stage):
        with torch.no_grad():
            z = torch.randn_like(action_shape_reference)
            proposal = self.generator.rollout(cond, z, steps=stage + 1)
            eps = torch.randn_like(proposal)
            t = self.sample_time(len(proposal), proposal.device)
            xt = (1 - t[:, None, None]) * eps + t[:, None, None] * proposal
            target = proposal - eps
        return F.mse_loss(self.fake_velocity(xt, t, cond, stage), target)

    def generator_loss(self, actions, cond, stage):
        z = torch.randn_like(actions)
        proposal = self.generator.rollout(cond, z, steps=stage + 1,
                                           detach_prefix=True)
        eps = torch.randn_like(proposal)
        t = self.sample_time(len(proposal), proposal.device)
        with torch.no_grad():
            xt = (1 - t[:, None, None]) * eps + t[:, None, None] * proposal.detach()
            vr = self.real_flow(xt, t, cond)
            vf = self.fake_velocity(xt, t, cond, stage)
            # x_t=(1-t)*eps+t*a; score_fake-score_real
            # = t/(1-t)*(v_fake-v_real).
            grad = ((t / (1. - t))[:, None, None] * (vf - vr))
            grad = torch.nan_to_num(grad).clamp(-100., 100.)
        return (proposal * grad).flatten(1).sum(1).mean()

    @torch.no_grad()
    def sample(self, cond, inference_steps=4, record_intermediate=False,
               clip_intermediate_actions=True):
        b = cond["state"].shape[0]
        z = torch.randn(b, self.horizon_steps, self.action_dim,
                        device=cond["state"].device)
        pred, chain = self.generator.rollout(cond, z, steps=inference_steps,
                                             return_chain=True)
        if clip_intermediate_actions:
            pred = pred.clamp(*self.act_range)
        return namedtuple("Sample", "trajectories chains")(
            pred, chain if record_intermediate else None)
