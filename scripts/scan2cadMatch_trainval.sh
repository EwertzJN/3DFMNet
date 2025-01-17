#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/experiment/scan2cad/miretr_raw"
export SNAPSHOT_DIR="${OUTPUT_DIR}/snapshots"
export LOG_DIR="${OUTPUT_DIR}/logs"
export PROPOSALS_ROOT="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/scan2cad_last_random_4096/"
export VOXEL_SIZE=0.025
# export TEST_EPOCH=24

python $PROJECT_ROOT/lib/scan2cadMatch_trainval.py \
    --output_dir $OUTPUT_DIR \
    --snapshot_dir $SNAPSHOT_DIR \
    --logs_dir $LOG_DIR \
    --voxel_size $VOXEL_SIZE \
    --proposals_dir $PROPOSALS_ROOT 