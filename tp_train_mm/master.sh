export MASTER_ADDR="192.168.123.199"        # 主节点的 IP 地址
export MASTER_PORT=2024               # 通信端口
export WORLD_SIZE=2                    # 总的进程数（2台机器，每台1个进程）
export RANK=0                          # 当前节点的 rank，主节点的 rank 是 0
export LOCAL_RANK=0  # 机器 1 上的 GPU 编号（此机器只有一个 GPU）

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="192.168.123.199" --master_port=12355 tp_mm.py


