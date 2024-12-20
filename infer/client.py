import requests
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# 服务器地址
server_url = "http://127.0.0.1:5000/generate"

# 加载数据集
dataset_path = "../dataset/ag_news_test.parquet"
df = pd.read_parquet(dataset_path)

# 提取文本列
texts = df["text"].tolist()

# 配置参数
rps = 10  # 请求每秒 (Requests Per Second)
duration = 10  # 测试持续时间（秒）

# 泊松间隔
intervals = np.random.poisson(1 / rps, int(rps * duration))

# 统计指标
throughput = 0
latency = []

# 发送请求并测量性能
start_time = time.time()
for i in tqdm(range(len(intervals)), desc="Sending Requests"):
    if i < len(texts):
        input_text = texts[i % len(texts)]
        payload = {"input_text": input_text}

        # 记录开始时间
        req_start_time = time.time()

        # 发送请求
        response = requests.post(server_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")
        else:
            generated_text = None

        # 记录结束时间
        req_end_time = time.time()

        # 计算时间
        latency.append(req_end_time - req_start_time)

        throughput += 1

    # 等待下一个泊松间隔
    if i < len(intervals) - 1:
        time.sleep(intervals[i])

end_time = time.time()

# 输出结果
print(f"Total Requests: {throughput}")
print(f"Throughput: {throughput / (end_time - start_time):.2f} requests/sec")
print(f"Average latency: {np.mean(latency):.2f} sec")
