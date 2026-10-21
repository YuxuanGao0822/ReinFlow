"""Train a Data-Space DMD policy from a trusted ReinFlow ReFlow checkpoint."""
import torch
from agent.pretrain.train_agent import PreTrainAgent


class TrainDataSpaceAgent(PreTrainAgent):
    def __init__(self, cfg):
        # A teacher checkpoint is not a PreTrainAgent training-state resume.
        super().__init__(cfg)
        if not cfg.get("teacher_checkpoint"):
            raise ValueError("Set teacher_checkpoint to a ReFlow BC checkpoint")
        self.model.initialize_from_reflow(cfg.teacher_checkpoint,
                                          use_ema=cfg.get("teacher_use_ema", True))
        self.ema_model.load_state_dict(self.model.state_dict())
        self.fake_optimizer = torch.optim.AdamW(
            list(self.model.fake_flow.parameters()) +
            list(self.model.fake_stage.parameters()),
            lr=cfg.train.get("fake_learning_rate", cfg.train.learning_rate),
            weight_decay=cfg.train.weight_decay)
        self.last_dmd_metrics = {}

    def get_loss(self, batch_data):
        actions, cond = batch_data
        stage = int(torch.randint(self.model.refinement_steps, (1,)).item())
        if torch.is_grad_enabled():
            fake_losses = []
            for _ in range(self.model.fake_updates):
                self.fake_optimizer.zero_grad(set_to_none=True)
                loss_fake = self.model.fake_loss(actions, cond, stage)
                loss_fake.backward()
                self.fake_optimizer.step()
                fake_losses.append(loss_fake.detach())
            self.fake_optimizer.zero_grad(set_to_none=True)
        else:
            fake_losses = []
        loss = self.model.generator_loss(actions, cond, stage)
        self.last_dmd_metrics = dict(stage=stage, fake_loss=(
            torch.stack(fake_losses).mean().item() if fake_losses else float("nan")))
        return loss

    def inference(self, cond):
        model = self.ema_model if self.test_model_type == "ema" else self.model
        return model.sample(cond, inference_steps=self.cfg.model.refinement_steps)

    def _save_fake_optimizer(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["fake_optimizer"] = self.fake_optimizer.state_dict()
        torch.save(payload, path)

    def save_model(self):
        super().save_model()
        self._save_fake_optimizer(f"{self.checkpoint_dir}/state_{self.epoch}.pt")

    def save_last_model(self):
        super().save_last_model()
        self._save_fake_optimizer(f"{self.checkpoint_dir}/last.pt")
