# DreamDojo — Agent Context (claude.md)

> Distilled context for AI agents working on this codebase. Read this first before exploring files.

## TL;DR

DreamDojo is an **action-conditioned Video2World model** (NOT text-to-video). It takes an initial camera frame + robot joint actions and predicts what the camera will see. Built on Cosmos-Predict2.5 (NVIDIA), uses a Wan2.1 VAE tokenizer and a DiT diffusion transformer. Available in 2B and 14B parameter variants.

## Current State (as of 2026-02-21)

### What's Set Up

- **Environment**: `.venv` (Python 3.10, PyTorch 2.7.0+cu128) symlinked to `../../envs/af-dreamdojo`
- **Models downloaded**: 2B GR1 (4.1 GB), 14B GR1 (28 GB), LAM (8 GB) — all converted to `model_ema_bf16.pt`
- **Dataset**: In-lab_Eval subset only (10 GR-1 tasks, 30 GB). Full training set NOT downloaded.
- **Eval complete**: Both 2B and 14B on `pnp_corn_robot` task (5 samples). Results in `results/`

### Eval Results

| Metric | 2B | 14B |
|--------|-----|------|
| PSNR | 22.35 | 23.09 |
| SSIM | 0.774 | 0.812 |
| LPIPS | 0.217 | 0.190 |

## File Map (read these in order if you need to understand the code)

| Priority | File | What it does |
|----------|------|-------------|
| 1 | `examples/action_conditioned.py` | Entry point for action-conditioned eval. Parses args, loads dataset, calls `inference()` |
| 2 | `cosmos_predict2/action_conditioned.py` | Core inference loop. Loads model, iterates dataset, runs autoregressive generation, computes metrics |
| 3 | `cosmos_predict2/action_conditioned_config.py` | CLI argument definitions (`ActionConditionedSetupArguments`, etc.) |
| 4 | `cosmos_predict2/config.py` | `ModelKey`, `MODEL_CHECKPOINTS` dict mapping model variants to HF checkpoint configs |
| 5 | `cosmos_predict2/experiments/base/action.py` | Experiment factory — reads `configs/*.yaml`, creates LazyDict experiment configs |
| 6 | `configs/2b_480_640_gr1.yaml` | YAML override for 2B GR-1 (action_dim, dataset_path, etc.) |
| 7 | `groot_dreams/dataloader.py` | `MultiVideoActionDataset` — dispatches to per-embodiment dataset loaders |
| 8 | `cosmos_predict2/_src/predict2/inference/video2world.py` | `Video2WorldInference` class — model loading + `generate_vid2world()` method |
| 9 | `cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py` | Wan2.1 VAE implementation (encode/decode video to/from 16-ch latent) |
| 10 | `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py` | HF/S3 checkpoint resolution. **PATCHED** — see below |

## Patches Applied (3 files modified from upstream)

These patches work around the gated `nvidia/Cosmos-Predict2.5-{2B,14B}` HuggingFace repos. Without access, the VAE and base checkpoint downloads fail with 403.

### 1. `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py`

**What changed**: Added `_GATED_REPO_FALLBACKS` dict mapping `"Wan2.1_VAE.pth"` to the locally cached copy from `Wan-AI/Wan2.1-T2V-1.3B`. Wrapped `get_checkpoint_path()` in try/except to gracefully handle gated repo errors.

**Why**: The Wan2.1 VAE is identical in both repos. Cosmos-Predict2.5 just bundles it as `tokenizer.pth`.

**Revert condition**: If you get access to `nvidia/Cosmos-Predict2.5-2B`, revert this file.

### 2. `cosmos_predict2/experiments/base/action.py`

**What changed**: Replaced two `load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)` calls with `load_path=""` (for both `_default_groot_config` and `_default_groot_config_14b`).

**Why**: These are training-only config values (path to pretrained base for fine-tuning init). They're never used during inference but they trigger gated repo downloads at import time.

### 3. `cosmos_predict2/_src/predict2/action/configs/.../exp_2B_action_conditioned_rectify_flow_gr00t.py`

**What changed**: Line 700: `load_path = get_checkpoint_path(...)` → `load_path = ""`

**Why**: Same as above — training config evaluated at import time.

## Missing Dependency: pytorch3d

`pytorch3d` fails to compile (CUDA 13.0 vs PyTorch's 12.8). Only `pytorch3d.transforms` is needed (pure Python, for rotation conversions in action normalization). Installed by copying the module directly from the pytorch3d git repo into site-packages.

If the environment is recreated, re-run:
```bash
cd /tmp && git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git
SITE=".venv/lib/python3.10/site-packages"
mkdir -p "$SITE/pytorch3d"
cp /tmp/pytorch3d/pytorch3d/__init__.py "$SITE/pytorch3d/"
cp -r /tmp/pytorch3d/pytorch3d/transforms "$SITE/pytorch3d/"
cp -r /tmp/pytorch3d/pytorch3d/common "$SITE/pytorch3d/"
```

Also install: `uv pip install h5py lerobot`

## Key Concepts

### Model Types

DreamDojo has three inference modes (defined in `cosmos_predict2/config.py:InferenceType`):
- `TEXT2WORLD` — text prompt → video (uses Cosmos base, not DreamDojo-specific)
- `IMAGE2WORLD` — single image → video
- `VIDEO2WORLD` — initial video + optional actions → future video (**this is what DreamDojo uses**)

### Experiment Naming

YAML files in `configs/` map to experiment names: `configs/2b_480_640_gr1.yaml` → `dreamdojo_2b_480_640_gr1`. The mapping is in `cosmos_predict2/experiments/base/action.py`.

### Checkpoint Resolution

`MODEL_CHECKPOINTS` in `config.py` maps `ModelKey(size, variant, distilled)` to `CheckpointConfig` objects. Each has:
- `.path` — triggers HF download (gated, may fail)
- `.experiment` — experiment name string
- `.s3.uri` — original S3 path (only works inside NVIDIA)

The `--checkpoint-path` CLI arg bypasses this by providing a local path directly.

### Action Conditioning Flow

```
Robot states (from dataset parquet)
    → Compute deltas (consecutive state differences)
    → Normalize per-joint (mean/std from metadata)
    → Rotation transforms (axis_angle → rotation_6d via pytorch3d)
    → Flatten to (T, action_dim) tensor
    → ActionEmbedder projects to (T, model_channels)
    → Injected into DiT transformer blocks
```

## Running Commands

### Quick eval (2B, 5 samples)

```bash
cd /workspace/arc_fabric/models/dreamdojo
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/action_conditioned.py \
  -o outputs/eval_2b_gr1 \
  --checkpoint-path checkpoints/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt \
  --checkpoints-dir checkpoints/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir results/eval_2b_gr1 \
  --num-frames 49 --num-samples 5 \
  --dataset-path "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/gr1_unified.pnp_corn_robot" \
  --data-split full --deterministic-uniform-sampling --disable-guardrails
```

### Quick eval (14B, 5 samples)

Same as above but replace:
- `--checkpoint-path checkpoints/14B_GR1_post-train/iter_000050000/model_ema_bf16.pt`
- `--checkpoints-dir checkpoints/14B_GR1_post-train`
- `--experiment dreamdojo_14b_480_640_gr1`
- `--save-dir results/eval_14b_gr1`
- `-o outputs/eval_14b_gr1`

### Import test

```bash
cd /workspace/arc_fabric/models/dreamdojo
PYTHONPATH=. .venv/bin/python -c "
from groot_dreams.dataloader import MultiVideoActionDataset; print('dataloader OK')
from cosmos_predict2.action_conditioned import inference; print('inference OK')
"
```

## Not Text-to-Video — Integration Notes

DreamDojo cannot be integrated into arc_fabric's standard T2V pipeline because:
1. Input is (frame + actions), not (text prompt)
2. Output is conditioned on specific robot joint trajectories
3. The text encoder is loaded but unused (empty prompt)

To integrate, you'd need a custom worker type (`workers/dreamdojo_worker.py`) that accepts:
```python
{
    "initial_frame": base64_image,
    "actions": [[float, ...], ...],  # (T, action_dim) list
    "num_frames": int,
    "model_size": "2b" | "14b",
}
```

And a corresponding UI panel for uploading initial frames and action sequences.

## Disk Usage

| Component | Size |
|-----------|------|
| `.venv` | 13 GB |
| `checkpoints/2B_GR1_post-train` | 4.1 GB |
| `checkpoints/14B_GR1_post-train` | 28 GB |
| `checkpoints/DreamDojo/LAM_400k.ckpt` | 8 GB |
| `datasets/` (In-lab_Eval only) | 30 GB |
| **Total** | **~83 GB** |

## Embodiments Available

| Embodiment | Config prefix | Dataset key | Notes |
|------------|--------------|-------------|-------|
| GR-1 | `*_gr1` | `gr1` | NVIDIA humanoid, action_dim=384 |
| Unitree G1 | `*_g1` | `g1` | Quadruped |
| AgiBot | `*_agibot` | `agibot` | |
| YAM | `*_yam` | `yam` | |

Only GR-1 checkpoints are currently downloaded. To use others, download from `nvidia/DreamDojo` and convert.
