"""CPU-only checks for the data-space generation, DMD, and PPO mathematics."""
import torch
from torch import nn
from model.data_space.iteration import DataSpaceGenerator
from model.data_space.dmd import DataSpaceDMD


class TinyVelocity(nn.Module):
    def __init__(self, h=4, a=2):
        super().__init__()
        self.linear = nn.Linear(h * a + 3, h * a)
        self.h, self.a = h, a

    def forward(self, x, t, cond):
        b = x.shape[0]
        features = torch.cat([x.reshape(b, -1), cond["state"].reshape(b, -1)], -1)
        return self.linear(features).reshape_as(x)


def test_one_step_checkpoint_equivalence():
    torch.manual_seed(0)
    v = TinyVelocity()
    model = DataSpaceGenerator(v, 4, 2, refinement_steps=3)
    z = torch.randn(5, 4, 2)
    c = {"state": torch.randn(5, 1, 3)}
    expected = z + v(z, torch.zeros(5), c)
    actual = model(torch.zeros_like(z), z, c, 0)
    torch.testing.assert_close(actual, expected)


def test_rollout_and_prefix_detachment():
    model = DataSpaceGenerator(TinyVelocity(), 4, 2, 3)
    z = torch.randn(5, 4, 2)
    c = {"state": torch.randn(5, 1, 3)}
    out, chain = model.rollout(c, z, steps=3, return_chain=True)
    assert out.shape == (5, 4, 2)
    assert chain.shape == (5, 4, 4, 2)
    torch.testing.assert_close(chain[:, 0], z)
    stage_3 = model.rollout(c, z, steps=3, detach_prefix=True)
    assert stage_3.requires_grad
    stage_3.square().mean().backward()
    assert model.velocity.linear.weight.grad is not None


def test_fake_and_generator_loss_have_finite_gradients():
    model = DataSpaceDMD(TinyVelocity(), "cpu", 4, 2, refinement_steps=3)
    a = torch.randn(6, 4, 2)
    c = {"state": torch.randn(6, 1, 3)}
    fl = model.fake_loss(a, c, 1)
    assert torch.isfinite(fl)
    fl.backward()
    assert model.fake_flow.linear.weight.grad is not None
    assert model.real_flow.linear.weight.grad is None
    model.zero_grad(set_to_none=True)
    # Force a controlled fake-vs-real disagreement for meaningful gradients.
    with torch.no_grad():
        model.fake_flow.linear.bias.add_(0.2)
    gl = model.generator_loss(a, c, 2)
    assert torch.isfinite(gl)
    gl.backward()
    assert model.generator.velocity.linear.weight.grad is not None
    assert model.fake_flow.linear.weight.grad is None


def test_stochastic_chain_logprob_consistency():
    # Import through the actual ReinFlow PPO integration; requires ReinFlow deps.
    from model.data_space.ppo import NoisyDataSpace, PPODataSpace
    gen = DataSpaceGenerator(TinyVelocity(), 4, 2, refinement_steps=3)
    policy = NoisyDataSpace(gen, 3, .05, .15)
    model = PPODataSpace.__new__(PPODataSpace)
    nn.Module.__init__(model)
    model.actor_ft = policy
    model.inference_steps = 3
    model.act_dim_total = 8
    model.device = "cpu"
    model.horizon_steps, model.action_dim = 4, 2
    model.act_min, model.act_max = -1., 1.
    c = {"state": torch.randn(5, 1, 3)}
    action, chain, logp = model.get_actions(
        c, eval_mode=False, save_chains=True,
        normalize_denoising_horizon=True,
        normalize_act_space_dimension=True)
    fresh, _, std = model.get_logprobs(
        c, chain, get_entropy=True, normalize_denoising_horizon=True,
        normalize_act_space_dimension=True)
    torch.testing.assert_close(logp, fresh.detach())
    assert chain.shape == (5, 4, 4, 2)
    assert action.abs().max() <= 1
    assert std.item() > 0
