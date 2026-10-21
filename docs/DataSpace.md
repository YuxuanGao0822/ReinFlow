# Data-Space Iterative Policy: experiment bootstrap

Research implementation on top of ReinFlow's **release** branch. Keep ReinFlow
ReFlow, Shortcut and ReinFlow-PPO results as matched baselines. This branch
has not yet been validated with the full Robomimic simulator / large checkpoints.

## Main components

- \`model/data_space/iteration.py\`: shared-source-noise refinement generator;
  for ReinFlow's orientation \`x_t=(1-t)z+t*a\`, initialization at \`t=0\` is
  \`z+velocity(z,0,condition)\`. New projection/embeddings are zero initialized.
- \`model/data_space/dmd.py\`: frozen BC real-flow teacher, online fake flow,
  stage-conditioned fake model, stop-grad self-rollout and conditional DMD.
- \`agent/pretrain/train_data_space_agent.py\`: pretraining entry point.
- \`model/data_space/ppo.py\`: data-space stochastic transitions and joint
  transition log probability; the source noise is stored at chain index 0.
- \`agent/finetune/reinflow/train_ppo_data_space_agent.py\`: reuse of ReinFlow
  rollout, buffer, GAE and PPO machinery.
- \`tests/test_data_space.py\`: algebraic/smoke checks.

**Research boundaries:** this first online RL version is *vanilla PPO on the
data-space refinement chain*, not a completed refinement-aware credit-assignment
algorithm. Do not report it as a second validated contribution until new
stage-specific advantage methods and controlled experiments are implemented.

## 0. Server setup

Do not use GPU 0; export e.g. \`CUDA_VISIBLE_DEVICES=1\` and set \`device=cuda:0\`.
CUDA_VISIBLE_DEVICES renumbers the selected physical GPU as logical cuda:0.
For multi-process multi-GPU experiments assign separate CUDA_VISIBLE_DEVICES
values and separate log directories, seeds and process configs.

Use ReinFlow's \`installation/reinflow-setup.md\` to construct the **original
compatible environment**. Verify \`python -c 'import torch, hydra, d4rl'\`
before starting, and verify the MuJoCo / Robomimic installation separately.

\`\`\`bash
cd /data/workspace/gaoyuxuan/ReinFlow
git status --short
git fetch origin
git switch feat/data-space-iteration-rl
git pull --ff-only origin feat/data-space-iteration-rl
export REINFLOW_DIR="$PWD"
export REINFLOW_DATA_DIR="$PWD/data"
export REINFLOW_LOG_DIR="$PWD/log"
export REINFLOW_WANDB_ENTITY="YOUR_WANDB_ENTITY"
mkdir -p "$REINFLOW_DATA_DIR" "$REINFLOW_LOG_DIR"
\`\`\`

## 1. Teacher checkpoint

Official ReinFlow mapping identifies the **Can-Image ReFlow** checkpoint as
\`robomimic/pretrain/can/ReFlow/state_2000.pt\`:
https://drive.google.com/file/d/1e2OwtcbJe_G2XcXLisEuJ-YFt7kJS-23/view

\`\`\`bash
python -m pip install gdown
mkdir -p "$REINFLOW_LOG_DIR/robomimic/pretrain/can/ReFlow"
gdown --id 1e2OwtcbJe_G2XcXLisEuJ-YFt7kJS-23 \
  -O "$REINFLOW_LOG_DIR/robomimic/pretrain/can/ReFlow/state_2000.pt"
\`\`\`

If checkpoint loading fails, inspect its keys *before modifying the architecture*:
\`torch.load(path, map_location="cpu", weights_only=True)\`. Expected top-level
keys include \`model\` and \`ema\`, and parameter keys with a \`network.\` prefix.

## 2. Dataset

The configured image dataset is
\`$REINFLOW_DATA_DIR/robomimic/can-img/train.npz\`.
The existing \`script/run.py\` will try to download it using ReinFlow's
official data Google Drive mapping if not found. Normalize statistics for PPO
are at \`$REINFLOW_DATA_DIR/robomimic/can-img/normalization.npz\`.

## 3. DMD pretraining (one GPU first)

\`\`\`bash
CUDA_VISIBLE_DEVICES=1 python script/run.py \
  --config-dir=cfg/robomimic/pretrain/can \
  --config-name=pre_data_space_mlp_img \
  device=cuda:0 \
  wandb.offline_mode=true
\`\`\`

Saved training checkpoint:
\`$REINFLOW_LOG_DIR/robomimic/pretrain/can/<hydra name>/<date>_<time>_<seed>/checkpoint/last.pt\`
(the actual path is printed by the trainer).

## 4. PPO, only after DMD rollout succeeds

\`\`\`bash
export DATA_SPACE_CKPT="/absolute/path/to/your/DMD/checkpoint/last.pt"
CUDA_VISIBLE_DEVICES=2 python script/run.py \
  --config-dir=cfg/robomimic/finetune/can \
  --config-name=ft_ppo_data_space_mlp_img \
  device=cuda:0 \
  wandb.offline_mode=true
\`\`\`

The first PPO update should yield new / old log-probability ratio approximately
1.0. For the image trainer note augmentation may affect this ratio; always
check the exact rollout, preprocessing, random crop and normalizer settings.

## 5. Checklist before expensive experiments

- Run \`pytest -q tests/test_data_space.py\` in the ReinFlow environment.
- Assert at initialization \`G(z,zero,c,0)==z+v_teacher(z,0,c)\` on a test batch.
- Check real flow remains frozen, fake flow receives gradients only from
  fake loss, and generator receives gradients only from DMD loss.
- Confirm 1,2,3,4-NFE DMD pretraining rollout works **before PPO**.
- Confirm exact conditional logprobs with stored source noise and raw
  refinement proposals; keep clipping outside density calculations.
- Run matching ReFlow / Shortcut and Flow-Renoise baseline seeds.
- Check genuine physical GPU usage with \`nvidia-smi\`, not only log strings.

## Known open issues

- Not yet benchmark-validated. The DMD fake/real distributional objective can
  collapse action modes or degrade closed-loop success.
- The pretrained ReFlow checkpoint must match the VisionFlowMLP configuration;
  strict \`state_dict\` loading intentionally fails on mismatch.
- Generic PreTrainAgent is designed around a single optimizer; the DMD trainer
  coordinates fake updates inside \`get_loss\`. Multi-optimizer resume and all
  mixed-precision/grad-accumulation combinations are not yet supported.
- PPO is currently **standard joint-chain PPO**. Stage-specific credit
  assignment / exploration baselines are still proposed research, not implemented.
- Separate experiments must use distinct output/log directories, and the
  pretraining script does not automatically download \`teacher_checkpoint\`.
