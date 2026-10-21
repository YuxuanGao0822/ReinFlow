"""Explicit Data-Space PPO entry points using established ReinFlow trainers.

These retain the original environment buffers, GAE, optimizer, and critic,
but replace the model's sampling/log-probability with PPODataSpace.
"""
from agent.finetune.reinflow.train_ppo_flow_agent import TrainPPOFlowAgent
from agent.finetune.reinflow.train_ppo_flow_img_agent import TrainPPOImgFlowAgent


class TrainPPODataSpaceAgent(TrainPPOFlowAgent):
    pass


class TrainPPOImgDataSpaceAgent(TrainPPOImgFlowAgent):
    pass
