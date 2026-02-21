# DreamDojo Architecture & Codebase Guide

## What DreamDojo Is

DreamDojo is an **action-conditioned Video2World model** built on top of NVIDIA's Cosmos-Predict2.5 foundation. It is **not** a text-to-video model. Given:

- An initial video frame (robot's ego camera view)
- A sequence of robot joint actions (ground truth or planned)

It predicts what the robot's camera will see in the future — i.e., it simulates the visual outcome of executing those actions.

**Primary use case**: Visual planning and model-predictive control for robotics. Before executing an action plan on a real robot, simulate it in DreamDojo to verify the outcome.

---

## Inputs & Outputs

### Inference Inputs

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| Initial frame | `(1, 3, 1, 480, 640)` | uint8 [0,255] | First RGB frame from robot ego camera |
| Robot actions | `(12, 384)` | float32 | 12 timesteps of robot state deltas. 384 = chunked multi-joint action vector for GR-1 |
| Prompt | `""` | string | Always empty string (no text conditioning) |
| Guidance | `0.0` | float | CFG guidance scale (typically 0 for action-conditioned) |
| Resolution | `"480,640"` | string | Output resolution |

### Inference Outputs

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| Predicted video | `(1, 3, 13, 480, 640)` | float32 [-1,1] | 13 predicted frames (1 conditional + 12 generated) |

### Action Format

Actions are **relative deltas** between consecutive robot states, not absolute positions. For GR-1:
- Each timestep has 384 dimensions (multiple joint groups concatenated)
- Joint groups: left_arm, right_arm, left_wrist, right_wrist, left_hand, right_hand, left_leg, right_leg, waist, neck, etc.
- The `groot_dreams/data/transform/state_action.py` handles normalization and rotation transforms

### Autoregressive Rollout

For long-horizon prediction (>12 timesteps), the model runs autoregressively:
1. Generate 13 frames from initial frame + 12 actions
2. Take the last generated frame as the new initial frame
3. Feed the next 12 actions
4. Repeat until all actions are consumed

---

## Repository Structure

```
models/dreamdojo/
├── examples/
│   ├── action_conditioned.py      # Action-conditioned eval entry point
│   └── inference.py               # General inference (text2world, image2world, video2world)
│
├── cosmos_predict2/               # Core model library (from Cosmos-Predict2.5)
│   ├── action_conditioned.py      #   Action-conditioned inference pipeline
│   ├── action_conditioned_config.py #   CLI argument definitions
│   ├── config.py                  #   ModelKey, MODEL_CHECKPOINTS registry
│   ├── inference.py               #   General inference CLI
│   ├── experiments/
│   │   └── base/action.py         #   Experiment config factory (loads YAML configs)
│   └── _src/
│       ├── predict2/
│       │   ├── action/            #   Action-conditioned model definitions
│       │   │   ├── models/        #     ActionVideo2WorldModel
│       │   │   ├── configs/       #     Hydra-style experiment configs
│       │   │   └── nets/          #     ActionConditionedMinimalV1LVGDiT
│       │   ├── inference/
│       │   │   └── video2world.py #   Video2WorldInference class
│       │   ├── tokenizers/
│       │   │   └── wan2pt1.py     #   Wan2.1 VAE (video tokenizer)
│       │   ├── text_encoders/     #   Cosmos-Reason1 text encoder
│       │   ├── nets/              #   DiT transformer architectures
│       │   └── utils/
│       │       └── model_loader.py #  Checkpoint loading
│       └── imaginaire/
│           └── utils/
│               └── checkpoint_db.py # HF/S3 checkpoint resolution
│
├── groot_dreams/                  # DreamDojo-specific data & training
│   ├── dataloader.py              #   MultiVideoActionDataset
│   ├── groot_configs.py           #   Modality configs per embodiment
│   ├── data/
│   │   ├── dataset.py             #   WrappedLeRobotSingleDataset
│   │   ├── dataset_mano.py        #   MANO hand dataset (EgoDex)
│   │   ├── dataset_video.py       #   Generic video dataset
│   │   └── transform/             #   Action normalization, rotation transforms
│   └── utils/                     #   LAM (Latent Action Model) utils
│
├── configs/                       # YAML overrides per embodiment/size
│   ├── 2b_480_640_gr1.yaml        #   2B GR-1 config
│   ├── 14b_480_640_gr1.yaml       #   14B GR-1 config
│   └── ...                        #   AgiBot, G1, YAM variants
│
├── scripts/
│   ├── convert_distcp_to_pt.py    # Convert distributed checkpoints → .pt
│   └── download_dataset.py        # Custom dataset downloader with retry
│
├── checkpoints/                   # Model weights (local)
├── datasets/                      # Evaluation datasets (local)
├── results/                       # Eval outputs
└── docs/                          # This documentation
```

---

## Model Architecture

### Pipeline Overview

```
Input Frame (480×640 RGB)
    │
    ▼
┌──────────────────┐
│  Wan2.1 VAE      │  Encoder: 3D causal conv, z_dim=16
│  (Video Tokenizer)│  Compression: 8× spatial, 4× temporal
└──────────────────┘
    │
    ▼
Latent (1, 16, 4, 60, 80)   ←── 16 channels, 4 temporal frames, 60×80 spatial
    │
    ▼
┌──────────────────────────────────────────────┐
│  DiT (Diffusion Transformer)                  │
│  + Action Conditioning via ActionEmbedder     │
│  + Rectified Flow scheduling                  │
│  35 denoising steps                           │
│                                               │
│  2B:  model_channels=2048, heads=16, blocks=28│
│  14B: model_channels=5120, heads=40, blocks=36│
└──────────────────────────────────────────────┘
    │
    ▼
Denoised Latent (1, 16, 4, 60, 80)
    │
    ▼
┌──────────────────┐
│  Wan2.1 VAE      │  Decoder: inverse of encoder
│  (Video Decoder) │
└──────────────────┘
    │
    ▼
Output Video (1, 3, 13, 480, 640)
```

### VAE Tokenizer (Wan2pt1)

- **Architecture**: 3D Causal Convolutional VAE (`Encoder3d` + `Decoder3d`)
- **Latent channels**: 16
- **Spatial compression**: 8× (480→60, 640→80)
- **Temporal compression**: 4× (13 frames → 4 latent frames)
- **Normalization**: Per-channel mean/std, shapes `(1, 16, 1, 1, 1)` for images, `(1, 16, 32, 1, 1)` for video
- **Source**: Shared with Wan2.1 (Alibaba's video model). The checkpoint `Wan2.1_VAE.pth` is identical in both `nvidia/Cosmos-Predict2.5-2B` and `Wan-AI/Wan2.1-T2V-1.3B`
- **File**: `cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py`

### Diffusion Model (DiT)

- **Architecture**: `MinimalV1LVGDiT` → `ActionConditionedMinimalV1LVGDiT`
- **Positional encoding**: RoPE3D (learnable)
- **Attention**: `minimal_a2a` backend
- **Patch size**: spatial=2, temporal=1
- **Conditioning**: AdaLN LoRA (dim=256)
- **Scheduler**: Rectified Flow (35 steps default)

| Parameter | 2B | 14B |
|-----------|-----|------|
| model_channels | 2048 | 5120 |
| num_heads | 16 | 40 |
| num_blocks | 28 | 36 |
| VRAM usage | ~58 GB | ~70 GB |
| Inference speed | ~8.4 it/s | ~1.8 it/s |

### Action Embedder

- Projects flattened action chunks into the transformer's hidden dimension
- `action_dim × num_action_per_chunk` → `model_channels`
- Default for GR-1: `action_dim=384`, `num_action_per_chunk=12`
- Zero-initialized at start of training (`zero_init_action_embedder=False` for post-training)

### Text Encoder (Cosmos-Reason1)

- **Model**: `nvidia/Cosmos-Reason1-7B` (based on Qwen2.5-VL-7B-Instruct)
- **Usage**: Generates text embeddings, but action-conditioned inference always uses `prompt=""` (empty string)
- **Note**: Still loaded during model init. Consumes ~14 GB VRAM
- **File**: `cosmos_predict2/_src/predict2/text_encoders/text_encoder.py`

### Latent Action Model (LAM)

- A separate model that generates "latent action" video representations
- Used by the 14B model for additional conditioning (`lam_video` input)
- Checkpoint: `checkpoints/DreamDojo/LAM_400k.ckpt` (8.5 GB)
- **File**: `groot_dreams/utils/`

---

## Experiment Config System

Configs are resolved in layers:

1. **Base experiment** (Hydra-style): `cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py`
2. **DreamDojo defaults**: `cosmos_predict2/experiments/base/action.py` — defines `_default_groot_config` and `_default_groot_config_14b`
3. **YAML overrides**: `configs/2b_480_640_gr1.yaml` — per-embodiment overrides for `action_dim`, `dataset_path`, etc.

The experiment name is constructed dynamically from YAML filenames:
- `configs/2b_480_640_gr1.yaml` → experiment name `dreamdojo_2b_480_640_gr1`
- `configs/14b_480_640_gr1.yaml` → experiment name `dreamdojo_14b_480_640_gr1`

This mapping happens in `cosmos_predict2/experiments/base/action.py` which glob-reads `configs/*.yaml` and registers each as a LazyDict experiment.

---

## Dataloader Architecture

```
MultiVideoActionDataset
    ├── VideoActionDataset (for LeRobot-format robot data: GR1, G1, AgiBot, YAM)
    │   └── WrappedLeRobotSingleDataset
    │       └── Reads parquet files + video files
    │       └── Applies modality transforms (normalization, rotation, action deltas)
    ├── MANODataset (for EgoDex hand data)
    └── VideoDataset (for generic video data)
```

Dataset type is inferred from the path string:
- Contains "gr1" → `embodiment="gr1"`, uses `VideoActionDataset`
- Contains "egodex_21" → uses `MANODataset`
- Otherwise → uses `VideoDataset`

Each dataset item returns:
```python
{
    "video": tensor (C, T, H, W),       # RGB video frames
    "lam_video": tensor,                  # LAM-encoded video
    "action": tensor (T-1, action_dim),   # Robot action deltas
    "fps": int,
    "num_frames": int,
    "padding_mask": tensor,
    "image_size": tuple,
    "ai_caption": str,
}
```

---

## Hybrid Model Feasibility

### Same Latent Space?

**Yes** — both 2B and 14B use the identical Wan2.1 VAE tokenizer with:
- 16-channel continuous latent space
- Same spatial (8×) and temporal (4×) compression
- Same mean/std normalization constants

### Same Architecture?

**Same family, different dimensions:**
- Both use `MinimalV1LVGDiT` (Diffusion Transformer)
- Same patch size, attention mechanism, RoPE3D positional encoding
- Different: `model_channels` (2048 vs 5120), `num_heads` (16 vs 40), `num_blocks` (28 vs 36)

### Can You Merge Them?

**Direct weight averaging: No** — dimension mismatch prevents element-wise operations on transformer blocks.

**Viable approaches:**
1. **Knowledge distillation**: Train 2B student with 14B teacher supervision
2. **Layer transplant**: Transfer dimension-compatible layers (input/output projections after interpolation)
3. **Ensemble**: Run both models and blend their latent predictions (expensive but straightforward)
4. **LoRA fine-tuning**: Train LoRA adapters on the 2B model using 14B-generated targets

---

## Key Gotchas

1. **Gated repos**: `nvidia/Cosmos-Predict2.5-{2B,14B}` require license acceptance. See ONBOARDING.md for workarounds.
2. **CUDA version mismatch**: System CUDA 13.0 vs PyTorch's 12.8 breaks pytorch3d compilation. Use transforms-only install.
3. **`get_checkpoint_path` at import time**: Several config files call `get_checkpoint_path()` at module import time, triggering HF downloads before any inference code runs. Must patch these to empty strings.
4. **LAM checkpoint**: 14B model silently fails without `checkpoints/DreamDojo/LAM_400k.ckpt`.
5. **GPU mapping**: `CUDA_VISIBLE_DEVICES` remaps GPU indices — device 0 inside the process may be physical GPU 1. Check `nvidia-smi` to verify actual placement.
6. **HF rate limits**: Downloading many small files (dataset) quickly hits the 1000 req/5min limit. Use `snapshot_download` with retry loops.
