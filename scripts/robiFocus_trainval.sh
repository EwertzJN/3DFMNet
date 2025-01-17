#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp1"
export SNAPSHOT_DIR="${OUTPUT_DIR}/snapshots"
export LOG_DIR="${OUTPUT_DIR}/logs"
export ROBI_ROOT="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/Robi/"

python $PROJECT_ROOT/lib/robiFocus_trainval.py \
    --output_dir $OUTPUT_DIR \
    --snapshot_dir $SNAPSHOT_DIR \
    --logs_dir $LOG_DIR \
    --robi_root $ROBI_ROOT 