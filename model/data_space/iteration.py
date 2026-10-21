"""Data-space iterative action generator for ReinFlow's time convention.

ReinFlow learns x_t=(1-t)*z+t*action and v=action-z, so the
checkpoint-compatible one-step map is z + v(z, t=0, condition).
"""
import torch
from torch import nn


class DataSpaceGenerator(nn.Module):
    def __init__(self, velocity: nn.Module, horizon_steps: int, action_dim: int,
                 refinement_steps: int = 4):
        super().__init__()
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be >= 1")
        self.velocity = velocity
        self.horizon_steps = int(horizon_steps)
        self.action_dim = int(action_dim)
        self.refinement_steps = int(refinement_steps)
        width = self.horizon_steps * self.action_dim
        self.previous_projection = nn.Linear(width, width, bias=False)
        self.stage_embedding = nn.Embedding(refinement_steps, width)
        nn.init.zeros_(self.previous_projection.weight)
        nn.init.zeros_(self.stage_embedding.weight)

    def forward(self, previous_action, source_noise, cond, stage: int):
        if not 0 <= stage < self.refinement_steps:
            raise ValueError("invalid refinement stage")
        if previous_action.shape != source_noise.shape:
            raise ValueError("previous action / source noise shape mismatch")
        b = source_noise.shape[0]
        if tuple(source_noise.shape[1:]) != (self.horizon_steps, self.action_dim):
            raise ValueError("invalid action chunk")
        ids = torch.full((b,), stage, device=source_noise.device, dtype=torch.long)
        delta = (self.previous_projection(previous_action.reshape(b, -1))
                 + self.stage_embedding(ids))
        network_input = source_noise + delta.reshape_as(source_noise)
        time = torch.zeros((b,), dtype=source_noise.dtype, device=source_noise.device)
        return source_noise + self.velocity(network_input, time, cond)

    def rollout(self, cond, source_noise, steps=None, detach_prefix=False,
                return_chain=False):
        steps = self.refinement_steps if steps is None else int(steps)
        if not 1 <= steps <= self.refinement_steps:
            raise ValueError("invalid number of refinement steps")
        proposal = torch.zeros_like(source_noise)
        chain = [source_noise] if return_chain else None
        for stage in range(steps):
            if detach_prefix and stage < steps - 1:
                with torch.no_grad():
                    proposal = self(proposal, source_noise, cond, stage)
            else:
                proposal = self(proposal, source_noise, cond, stage)
            if return_chain:
                chain.append(proposal)
        if return_chain:
            return proposal, torch.stack(chain, dim=1)
        return proposal

    @torch.no_grad()
    def sample_action(self, cond, inference_steps=None,
                      clip_intermediate_actions=True, act_range=(-1., 1.),
                      z=None, save_chains=False):
        b = cond["state"].shape[0]
        if z is None:
            z = torch.randn(b, self.horizon_steps, self.action_dim,
                            device=cond["state"].device)
        pred, chain = self.rollout(cond, z, steps=inference_steps,
                                    return_chain=True)
        if clip_intermediate_actions:
            pred = pred.clamp(*act_range)
        return (pred, chain) if save_chains else pred
