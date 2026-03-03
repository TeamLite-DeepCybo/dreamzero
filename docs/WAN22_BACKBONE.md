# Training DreamZero with Wan2.2-TI2V-5B Backbone

This guide explains how to train DreamZero on the DROID dataset using **Wan2.2-TI2V-5B** as the backbone instead of the default Wan2.1-I2V-14B.

## Architecture Differences

| Component | Wan2.1-I2V-14B | Wan2.2-TI2V-5B |
|-----------|-----------------|----------------|
| DiT dim | 2048 | 3072 |
| DiT layers | 32 | 30 |
| DiT heads | 16 | 24 |
| FFN dim | 8192 | 14336 |
| VAE latent channels | 16 | 48 |
| VAE spatial stride | 8× | 16× |
| Model type | i2v | ti2v |

DreamZero uses a **CausalWanModel** wrapper that extends the base Wan architecture with action/state registers for robot policy learning. The same `CausalWanModel` class supports both Wan2.1 and Wan2.2 backbones via configuration—no new class is required. The config switches the architecture parameters (dim, in_dim, out_dim, etc.) and uses `WanVideoVAE38` for the 48-channel Wan2.2 VAE.

## Prerequisites

1. **Wan2.2-TI2V-5B** weights:
   ```bash
   huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B
   ```
   Or clone from [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2) and follow their download instructions.

2. **Image encoder (CLIP)**: Wan2.2-TI2V-5B does not include the CLIP image encoder. Use the one from Wan2.1:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
   ```
   Only `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` is needed.

3. **DROID dataset** in LeRobot format:
   ```bash
   huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
   ```

## Quick Start

```bash
# Set paths (optional - defaults shown)
export WAN22_CKPT_DIR=./checkpoints/Wan2.2-TI2V-5B
export IMAGE_ENCODER_DIR=./checkpoints/Wan2.1-I2V-14B-480P  # for CLIP only
export DROID_DATA_ROOT=./data/droid_lerobot

# Run training
bash scripts/train/droid_training_wan22.sh
```

## Configuration Details

The Wan2.2 config (`wan_flow_matching_action_tf_wan22.yaml`) overrides:

- **model/dreamzero/action_head**: `wan_flow_matching_action_tf_wan22`
- **diffusion_model_cfg**: Wan2.2 architecture (dim=3072, in_dim=48, out_dim=48, etc.)
- **vae_cfg**: `WanVideoVAE38` (48-channel Wan2.2 VAE)
- **frame_seqlen**: 55 (for 320×176 resolution with 16× spatial compression)

For other resolutions, adjust `frame_seqlen`:
- 320×176: 55
- 640×352: 220

## Using with Custom Training Scripts

To use Wan2.2 in your own training script, add:

```bash
model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
dit_version=$WAN22_CKPT_DIR \
text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
frame_seqlen=55
```

## File Layout

```
dreamzero/
├── groot/vla/configs/model/dreamzero/action_head/
│   ├── wan_flow_matching_action_tf.yaml      # Wan2.1 (default)
│   └── wan_flow_matching_action_tf_wan22.yaml  # Wan2.2-TI2V-5B
├── scripts/train/
│   ├── droid_training.sh           # Wan2.1 backbone
│   └── droid_training_wan22.sh     # Wan2.2 backbone
└── docs/
    └── WAN22_BACKBONE.md          # This file
```

The action head (`wan_flow_matching_action_tf.py`) automatically detects Wan2.2 vs Wan2.1 based on `in_dim` (48 vs 16) and `vae.z_dim` (48 vs 16), and loads the correct checkpoint files from the appropriate HuggingFace repos when local paths are not found.
