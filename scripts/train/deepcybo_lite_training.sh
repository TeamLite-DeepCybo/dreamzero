#!/bin/bash
# DreamZero DeepCybo-Lite Training Script (single RTX PRO 6000, LoRA post-training)
#
# Post-trains the DreamZero-AgiBot checkpoint on the deepcybo_lite_bilateral
# pick-and-place dataset (150 episodes, 16-dim state/action, 3 views).
export HYDRA_FULL_ERROR=1
export PATH=/home/admin/fbfm_ws/envs/dreamzero/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=${HOME}/dreamzero:$PYTHONPATH

# ============ CONFIGURATION ============
WS=/mnt/data16t/dreamzero_deepcybo
DATA_ROOT=${DATA_ROOT:-"$WS/datasets/deepcybo_lite_bilateral_gear"}
OUTPUT_DIR=${OUTPUT_DIR:-"$WS/runs/dreamzero_deepcybo_lora_v1"}
PRETRAINED=${PRETRAINED:-"$WS/checkpoints/DreamZero-AgiBot"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$WS/checkpoints/umt5-xxl-tokenizer"}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
export WANDB_DIR="$WS/cache/wandb"
export HF_HOME="$WS/cache/huggingface"
mkdir -p "$WANDB_DIR"
export WANDB_MODE=${WANDB_MODE:-offline}
# =======================================

if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$PRETRAINED/model.safetensors.index.json" ]; then
    echo "ERROR: DreamZero-AgiBot checkpoint incomplete at $PRETRAINED"
    exit 1
fi

cd "$(dirname "$0")/../.."

python -m torch.distributed.run --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/deepcybo_lite_relative \
    wandb_project=dreamzero_deepcybo \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=2500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$BATCH_SIZE \
    training_args.gradient_accumulation_steps=2 \
    max_steps=20000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    deepcybo_lite_data_root=$DATA_ROOT \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
