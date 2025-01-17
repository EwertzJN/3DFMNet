#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="2"

export TEST_EPOCH=46
export IF_PROPOSAL=True
export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/scan2cad/detection_exp0"
export SNAPSHOT_DIR="${OUTPUT_DIR}/snapshots"
export LOG_DIR="${OUTPUT_DIR}/logs"
export PROPOSAL_DIR="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/scan2cad_last_random_4096/"
export SCAN2CAD_ROOT="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/scan2cad/scan2cad_pre/"

python $PROJECT_ROOT/lib/scan2cadFocus_test.py \
    --test_epoch $TEST_EPOCH \
    --if_proposal $IF_PROPOSAL \
    --output_dir $OUTPUT_DIR \
    --snapshot_dir $SNAPSHOT_DIR \
    --logs_dir $LOG_DIR \
    --proposals_dir $PROPOSAL_DIR \
    --scan2cad_root $SCAN2CAD_ROOT 