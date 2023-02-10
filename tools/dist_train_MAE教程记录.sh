#!/usr/bin/env bash

CONFIG=$1  # 需要传输的第一个参数，即配置文件的路径
GPUS=$2  # 需要传输的第二个参数，即要使用的GPU格式个数
NNODES=${NNODES:-1}  # 所有的结点数（这里默认的是1，即只使用一个结点，即单机）
NODE_RANK=${NODE_RANK:-0}  # 结点编号（因为编号是从0开始，而且只有一个结点，所以这里是0）
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \  # 所有的结点总数
    --node_rank=$NODE_RANK \  # 每个结点对应的序号，例如：0或1或2。
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \  # 每个结点开启的进程数。
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}  # 使用的环境，这里使用pytorch

# # 将下面的命令输入bash，也可正常运行
# python -m torch.distributed.launch \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=1 \
#     tools/train.py \
#     configs/selfsup/mae/mae_vit-base-p16_1xb32-coslr-1e_tinyin200.py \
#     --seed 0 \
#     --launcher pytorch --work_dir work_dirs/selfsup/mae/mae_vit-base-p16_1xb32-coslr-1e_tinyin200
