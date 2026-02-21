# DreamDojo Onboarding Guide

## Environment Setup

### Python Environment

DreamDojo uses `uv` (not conda) with Python 3.10 and CUDA 12.8. The environment lives at `models/dreamdojo/.venv`.

```bash
cd /workspace/arc_fabric/models/dreamdojo
bash install.sh
```

The `install.sh` creates a `.venv` via `uv` and installs all deps. After setup, the env is symlinked into arc_fabric's env directory:

```bash
ln -s /workspace/arc_fabric/models/dreamdojo/.venv /workspace/arc_fabric/envs/af-dreamdojo
```

This lets arc_fabric's `worker_manager` use `conda run --prefix /workspace/arc_fabric/envs/af-dreamdojo` to invoke the env (it works because `conda run --prefix` just needs a Python environment at that path).

### Missing Dependencies (not in install.sh)

The following packages are required but not installed by `install.sh`:

```bash
# pytorch3d - only the transforms module is needed (pure Python)
# Full pytorch3d fails to compile due to CUDA 13.0 vs 12.8 mismatch
# Workaround: clone pytorch3d and copy just the transforms module
cd /tmp && git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git pytorch3d_src
SITE="/workspace/arc_fabric/models/dreamdojo/.venv/lib/python3.10/site-packages"
cp /tmp/pytorch3d_src/pytorch3d/__init__.py "$SITE/pytorch3d/"
cp -r /tmp/pytorch3d_src/pytorch3d/transforms "$SITE/pytorch3d/"
cp -r /tmp/pytorch3d_src/pytorch3d/common "$SITE/pytorch3d/"

# h5py and lerobot - needed by the dataloader
uv pip install h5py lerobot
```

### Version Info

| Component | Version |
|-----------|---------|
| Python | 3.10.19 |
| PyTorch | 2.7.0+cu128 |
| CUDA (torch) | 12.8 |
| System CUDA | 13.0 |

The CUDA 13.0 vs 12.8 mismatch causes pytorch3d compilation to fail (torch's `cpp_extension._check_cuda_version` raises). The transforms-only workaround above avoids this entirely since `pytorch3d.transforms` is pure Python.

---

## Model Checkpoints

### Downloading from HuggingFace

Models are hosted at `nvidia/DreamDojo` on HuggingFace. They come in **distributed checkpoint** (`.distcp`) format and must be converted to standard PyTorch `.pt` files.

```bash
# Download (example for 2B GR1)
cd /workspace/arc_fabric/models/dreamdojo
PYTHONPATH=. .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/DreamDojo', allow_patterns='2B_GR1_post-train/**', local_dir='checkpoints')
"
```

**Rate limiting**: HuggingFace enforces 1000 API requests per 5 minutes. For large downloads, wrap in a retry loop that catches `HfHubHTTPError` with "429" and sleeps for 310 seconds before retrying. `snapshot_download` resumes automatically.

### Converting Distributed Checkpoints

After downloading, convert `.distcp` to `.pt`:

```bash
PYTHONPATH=. .venv/bin/python scripts/convert_distcp_to_pt.py \
  --checkpoint-dir checkpoints/2B_GR1_post-train/iter_000050000
```

This produces:
- `model.pt` (full checkpoint)
- `model_ema_fp32.pt` (EMA weights, fp32)
- `model_ema_bf16.pt` (EMA weights, bf16 — **this is what inference uses**)

After conversion, you can delete `model.pt` and `model_ema_fp32.pt` to save disk space (~80% reduction).

### LAM Checkpoint

The 14B model requires the Latent Action Model checkpoint:

```bash
PYTHONPATH=. .venv/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('nvidia/DreamDojo', 'LAM_400k.ckpt', local_dir='checkpoints/DreamDojo')
"
```

### Current Checkpoints on Disk

| Checkpoint | Path | Size |
|------------|------|------|
| 2B GR1 post-train | `checkpoints/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt` | 4.1 GB |
| 14B GR1 post-train | `checkpoints/14B_GR1_post-train/iter_000050000/model_ema_bf16.pt` | 28 GB |
| LAM 400k | `checkpoints/DreamDojo/LAM_400k.ckpt` | 8.0 GB |

### Available Model Configs

All configs are in `configs/` and follow the pattern `{size}_{resolution}_{embodiment}.yaml`:

| Config | Embodiment | Notes |
|--------|------------|-------|
| `2b_480_640_gr1.yaml` | GR-1 robot | action_dim=384, 12 actions per chunk |
| `14b_480_640_gr1.yaml` | GR-1 robot | action_dim=384, 12 actions per chunk |
| `2b_480_640_g1.yaml` | Unitree G1 | Different action space |
| `14b_480_640_g1.yaml` | Unitree G1 | |
| `2b_480_640_agibot.yaml` | AgiBot | |
| `14b_480_640_agibot.yaml` | AgiBot | |
| `2b_480_640_yam.yaml` | YAM | |
| `14b_480_640_yam.yaml` | YAM | |
| `*_pretrain.yaml` | N/A | Pretraining configs |

---

## Dataset

### Source

Datasets are hosted at `nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1` on HuggingFace.

### What's Downloaded

We downloaded only the **In-lab_Eval** subset (10 tasks, ~30 GB total). The full GR1_robot training set has 44k files and is impractical to download due to HF rate limits.

```
datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/
├── gr1_unified.color_puzzles_robot/     (1576 files)
├── gr1_unified.fold_cloth_long_robot/   (524 files)
├── gr1_unified.fold_cloth_robot/        (208 files)
├── gr1_unified.mug_robot/              (212 files)
├── gr1_unified.pnp_corn_robot/         (110 files)  ← used for eval
├── gr1_unified.pnp_cucumber_robot/     (112 files)
├── gr1_unified.pnp_dragonfruit_robot/  (206 files)
├── gr1_unified.pnp_handover_plate_robot/ (635 files)
├── gr1_unified.pour_items_into_basket_robot/ (208 files)
└── gr1_unified.right_to_left_handover_corn_robot/ (170 files)
```

Each task directory follows the LeRobot format:
```
gr1_unified.pnp_corn_robot/
├── data/chunk-000/          # Parquet files with robot states per episode
│   ├── episode_000000.parquet
│   └── ...
├── meta/                    # Dataset metadata
└── videos/chunk-000/        # Video files per episode
    └── observation.images.ego_view_freq20/
```

### Downloading Specific Subsets

```python
from huggingface_hub import snapshot_download
snapshot_download(
    "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
    allow_patterns="In-lab_Eval/gr1_unified.pnp_corn_robot/**",
    local_dir="datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1",
)
```

---

## Gated Repository Workaround

DreamDojo depends on `nvidia/Cosmos-Predict2.5-2B` (and 14B) which are **gated repos** requiring license acceptance at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B. Without access, the following components fail to download:

1. **VAE tokenizer** (`tokenizer.pth`) — the Wan2.1 video VAE
2. **Base model checkpoints** — pre-trained weights referenced in experiment configs

### Patches Applied

Three files were patched to work around the gated repo:

**1. `cosmos_predict2/_src/imaginaire/utils/checkpoint_db.py`**

Added a fallback map `_GATED_REPO_FALLBACKS` that redirects the Wan2.1 VAE to a non-gated source (`Wan-AI/Wan2.1-T2V-1.3B`). Also wrapped `get_checkpoint_path` in try/except to return empty string on gated repo errors instead of crashing.

The Wan2.1 VAE is identical — Cosmos-Predict2.5 uses it as its video tokenizer. Download it once:

```python
from huggingface_hub import hf_hub_download
hf_hub_download('Wan-AI/Wan2.1-T2V-1.3B', 'Wan2.1_VAE.pth')
# Cached at: /workspace/.hf_home/hub/models--Wan-AI--Wan2.1-T2V-1.3B/.../Wan2.1_VAE.pth
```

**2. `cosmos_predict2/experiments/base/action.py`**

Replaced `load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)` with `load_path=""` for both 2B and 14B default configs. These `load_path` values are only used during training (to initialize from the pretrained base), not during inference.

**3. `cosmos_predict2/_src/predict2/action/configs/.../exp_2B_action_conditioned_rectify_flow_gr00t.py`**

Same fix — replaced `load_path = get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri)` with `load_path = ""`.

### If You Get Gated Repo Access

If you accept the license at https://huggingface.co/nvidia/Cosmos-Predict2.5-2B, you can revert all patches. The code will download components directly from the official repos.

---

## Running Evaluation

### 2B Model

```bash
cd /workspace/arc_fabric/models/dreamdojo
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/action_conditioned.py \
  -o outputs/eval_2b_gr1 \
  --checkpoint-path checkpoints/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt \
  --checkpoints-dir checkpoints/2B_GR1_post-train \
  --experiment dreamdojo_2b_480_640_gr1 \
  --save-dir results/eval_2b_gr1 \
  --num-frames 49 \
  --num-samples 5 \
  --dataset-path "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/gr1_unified.pnp_corn_robot" \
  --data-split full \
  --deterministic-uniform-sampling \
  --disable-guardrails
```

### 14B Model

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 .venv/bin/python examples/action_conditioned.py \
  -o outputs/eval_14b_gr1 \
  --checkpoint-path checkpoints/14B_GR1_post-train/iter_000050000/model_ema_bf16.pt \
  --checkpoints-dir checkpoints/14B_GR1_post-train \
  --experiment dreamdojo_14b_480_640_gr1 \
  --save-dir results/eval_14b_gr1 \
  --num-frames 49 \
  --num-samples 5 \
  --dataset-path "datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/gr1_unified.pnp_corn_robot" \
  --data-split full \
  --deterministic-uniform-sampling \
  --disable-guardrails
```

### GPU Requirements

- **2B model**: ~58 GB VRAM (fits on single H100 80GB). Inference speed: ~8.4 it/s.
- **14B model**: ~70 GB VRAM (fits on single H100 80GB). Inference speed: ~1.8 it/s.
- Both models can run in parallel on 2x H100s.
- The `--checkpoint-path` flag is required to bypass gated repo resolution of the base model.

### Eval Results (5 samples, `pnp_corn_robot`)

| Metric | 2B GR1 | 14B GR1 |
|--------|--------|---------|
| PSNR | 22.350 | 23.086 |
| SSIM | 0.774 | 0.812 |
| LPIPS | 0.217 | 0.190 |

### Output Files

Results are saved to `results/eval_{model}_gr1/iter_XXXXXXXXX/`:
- `XXXX_pred.mp4` — model's predicted video
- `XXXX_gt.mp4` — ground truth video
- `XXXX_merged.mp4` — side-by-side comparison (GT left, pred right)
- `XXXX_actions.npy` — the action sequence used
- `XXXX_metrics.json` — per-sample PSNR/SSIM/LPIPS
- `all_summary.json` — averaged metrics across all samples
