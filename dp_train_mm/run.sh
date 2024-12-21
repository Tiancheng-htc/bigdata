slaver="b@192.168.123.47"

./master.sh &

ssh $slaver "source ~/miniconda3/etc/profile.d/conda.sh && conda activate vllm && cd /home/b/bigdata/train && ./slaver.sh"