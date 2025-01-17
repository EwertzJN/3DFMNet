#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="3"

export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/experiment/robi/withoverlapmask_finetune"
# export OUTPUT_DIR="/data1/local_userdata/zhangliyuan/log/NIPS24/baseline/robi/exp13"
export SNAPSHOT_DIR="${OUTPUT_DIR}/snapshots"
export LOG_DIR="${OUTPUT_DIR}/logs"
export PROPOSALS_ROOT="/data1/local_userdata/zhangliyuan/dataset/NIPS2024/dataset_generate/robi/test/ours_datasets_1.5/"
export VOXEL_SIZE=0.00055
export TEST_EPOCH=41

python $PROJECT_ROOT/lib/robiMatch_test.py \
    --test_epoch $TEST_EPOCH \
    --output_dir $OUTPUT_DIR \
    --snapshot_dir $SNAPSHOT_DIR \
    --logs_dir $LOG_DIR \
    --voxel_size $VOXEL_SIZE \
    --proposals_dir $PROPOSALS_ROOT 