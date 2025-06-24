#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
declare -i START_EPOCH=$4
declare -i END_EPOCH=$5
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for i in $(seq $START_EPOCH $END_EPOCH)
do
    echo "$CHECKPOINT/epoch_$i.pth"
    if test -f $CHECKPOINT/epoch_$i.pth; then
        echo "Checkpoint of epoch $i exists."
        PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
        python -m torch.distributed.launch \
            --nnodes=$NNODES \
            --node_rank=$NODE_RANK \
            --master_addr=$MASTER_ADDR \
            --nproc_per_node=$GPUS \
            --master_port=$PORT \
            $(dirname "$0")/test.py \
            $CONFIG \
            $CHECKPOINT/epoch_$i.pth \
            --launcher pytorch \
            ${@:6}
        sleep 5
    fi
done

