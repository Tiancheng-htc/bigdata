export MASTER_ADDR="192.168.123.134"  # 机器 1 作为主节点
export MASTER_PORT="2024"
export WORLD_SIZE=2  # 总进程数为 2
export RANK=1  # 机器 2 上的进程编号
export LOCAL_RANK=0  # 机器 2 上的 GPU 编号（此机器只有一个 GPU）

torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr="192.168.123.134" --master_port=2024 data_parallel.py
