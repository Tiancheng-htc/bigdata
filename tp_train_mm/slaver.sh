export MASTER_ADDR="192.168.123.134"        # 主节点的 IP 地址（即机器 1）
export MASTER_PORT=12355               # 通信端口
export WORLD_SIZE=2                    # 总的进程数（2台机器，每台1个进程）
export RANK=1                          # 当前节点的 rank，工作节点的 rank 是 1

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.123.134" --master_port=12355 tp_mm.py

