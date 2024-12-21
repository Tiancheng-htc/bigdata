export MASTER_ADDR="192.168.123.199"        # 主节点的 IP 地址（即机器 1）
export MASTER_PORT=12345               # 通信端口
export WORLD_SIZE=2                    # 总的进程数（2台机器，每台1个进程）
export RANK=1                          # 当前节点的 rank，工作节点的 rank 是 1
export LOCAL_RANK=0  # 机器 1 上的 GPU 编号（此机器只有一个 GPU）

torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="192.168.123.199" --master_port=12355 tp_mm.py


