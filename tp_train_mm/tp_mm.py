import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import time
from datasets import load_dataset
import os

# 初始化分布式环境
def init_distributed():
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://192.168.123.199:2024",  # 主节点的 IP 和端口
        world_size=int(os.environ["WORLD_SIZE"]), 
        rank=int(os.environ["RANK"])
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx])

# 动态填充的 collate_fn
def collate_fn(batch):
    batch = [item.tolist() for item in batch]
    max_len = max(len(item) for item in batch)
    padded = [item + [0] * (max_len - len(item)) for item in batch]
    return torch.tensor(padded)

# 设置超参数
embed_size = 1024
heads = 16
num_layers = 24
vocab_size = 50257
max_length = 1024
dropout = 0.1
expansion_factor = 4
batch_size = 16
num_epochs = 1
learning_rate = 3e-5

# 加载数据集
dataset = load_dataset("wangrongsheng/ag_news")

# 数据预处理
def preprocess_data(data):
    tokenizer = lambda text: [ord(c) for c in text]  # 简单的字符级 tokenizer
    return [tokenizer(item['text'])[:max_length] for item in data]

train_tokens = preprocess_data(dataset['train'])
test_tokens = preprocess_data(dataset['test'])

# 创建自定义数据集
train_dataset = CustomDataset(train_tokens)
test_dataset = CustomDataset(test_tokens)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 自定义 GPT-2 网络结构
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        # 权重矩阵分配到两个设备（按列分割）
        self.values = nn.Linear(embed_size // 2, embed_size // 2, bias=False)  # 左侧设备
        self.keys = nn.Linear(embed_size // 2, embed_size // 2, bias=False)    # 右侧设备
        self.queries = nn.Linear(embed_size // 2, embed_size // 2, bias=False) # 右侧设备
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        left_input, right_input = torch.chunk(x, 2, dim=-1)  # 按列分割输入

        # 左侧设备计算
        left_values = self.values(left_input)
        left_keys = self.keys(left_input)
        left_queries = self.queries(left_input)

        # 右侧设备计算
        right_values = self.values(right_input)
        right_keys = self.keys(right_input)
        right_queries = self.queries(right_input)

        # 跨设备通信，使用 all_reduce 来同步查询、键和值的计算
        dist.all_reduce(left_values, op=dist.ReduceOp.SUM)
        dist.all_reduce(left_keys, op=dist.ReduceOp.SUM)
        dist.all_reduce(left_queries, op=dist.ReduceOp.SUM)

        dist.all_reduce(right_values, op=dist.ReduceOp.SUM)
        dist.all_reduce(right_keys, op=dist.ReduceOp.SUM)
        dist.all_reduce(right_queries, op=dist.ReduceOp.SUM)

        # Attention 计算（跨设备同步后的数据）
        energy = torch.einsum("nqhd,nkhd->nqkh", [left_queries, left_keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nqkh,nvhd->nqhd", [attention, left_values])

        # 输出
        return self.fc_out(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion_factor):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, expansion_factor * embed_size)
        self.fc2 = nn.Linear(expansion_factor * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, embed_size, heads, dropout, expansion_factor):
        super(Block, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, expansion_factor)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.attention(x)
        x = self.dropout1(self.norm1(attention + x))
        forward = self.feed_forward(x)
        x = self.dropout2(self.norm2(forward + x))
        return x

class GPT2(nn.Module):
    def __init__(self, embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor):
        super(GPT2, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [Block(embed_size, heads, dropout, expansion_factor) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        out = self.word_embeddings(x) + self.position_embeddings(positions)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out)

        return self.fc_out(out)

# 初始化分布式环境
init_distributed()

# 初始化模型并移动到对应 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2(embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

# 训练模型
for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    for i, batch in enumerate(train_loader):
        batch_start_time = time.time()
        batch = batch.to(device).long()

        optimizer.zero_grad()
        with autocast():
            output = model(batch)
            loss = loss_fn(output.view(-1, vocab_size), batch.view(-1))
        
        # 反向传播和优化步骤
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_end_time = time.time()
        print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss.item()}")
    
    epoch_end_time = time.time()
    print(f"Epoch {epoch + 1} completed in {(epoch_end_time - epoch_start_time):.4f}s")
