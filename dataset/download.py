from datasets import load_dataset
import pandas as pd

# 下载数据集
dataset = load_dataset("wangrongsheng/ag_news")

# 保存为 Parquet 文件
dataset['train'].to_parquet('ag_news_train.parquet')
dataset['test'].to_parquet('ag_news_test.parquet')

# 解析 Parquet 文件
train_df = pd.read_parquet('ag_news_train.parquet')
test_df = pd.read_parquet('ag_news_test.parquet')

