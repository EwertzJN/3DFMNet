#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="1"

export TEST_EPOCH=60
export IF_PROPOSAL=False
export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/detection_exp0"
export SNAPSHOT_DIR="${OUTPUT_DIR}/snapshots"
export LOG_DIR="${OUTPUT_DIR}/logs"
export DETCENTER_DIR="${OUTPUT_DIR}/detcenters"
export ROBI_ROOT="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/Robi/"

python $PROJECT_ROOT/lib/robiFocus_test.py \
    --test_epoch $TEST_EPOCH \
    --if_proposal $IF_PROPOSAL \
    --output_dir $OUTPUT_DIR \
    --snapshot_dir $SNAPSHOT_DIR \
    --logs_dir $LOG_DIR \
    --detcenter_dir $DETCENTER_DIR \
    --robi_root $ROBI_ROOT 