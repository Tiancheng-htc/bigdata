import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import argparse
from datasets import load_dataset

# 初始化分布式环境
def init_distributed_mode(args):
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)

# 自定义 GPT-2 网络结构，分割层
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, device1, device2):
        super(MultiHeadSelfAttention, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        # 将不同的张量操作分配到不同的设备上
        self.values = nn.Linear(embed_size, embed_size, bias=False).to(self.device1)
        self.keys = nn.Linear(embed_size, embed_size, bias=False).to(self.device2)
        self.queries = nn.Linear(embed_size, embed_size, bias=False).to(self.device2)
        self.fc_out = nn.Linear(embed_size, embed_size).to(self.device2)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)  # 在device1上
        keys = self.keys(x)  # 在device2上
        queries = self.queries(x)  # 在device2上

        # 张量分割：将张量切分到不同的设备上
        values = values.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3).to(self.device1)  # device1
        keys = keys.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3).to(self.device2)  # device2
        queries = queries.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3).to(self.device2)  # device2

        # 计算注意力：keys 和 queries 在 device2 上，values 在 device1 上
        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])  # 计算注意力分数，device2
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)  # softmax 在 device2
        out = torch.einsum("nqkh,nvhd->nqhd", [attention, values])  # 张量分割：将 attention 和 values 在 device1 上计算
        out = out.reshape(N, seq_length, self.embed_size)

        # 跨节点同步：使用 all_reduce 合并所有节点上的结果
        dist.all_reduce(out, op=dist.ReduceOp.SUM)  # 跨节点通信：汇总张量

        out = self.fc_out(out)  # 最后全连接层在 device2 上
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size, expansion_factor):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, expansion_factor * embed_size)
        self.fc2 = nn.Linear(expansion_factor * embed_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, embed_size, heads, dropout, expansion_factor, device1, device2):
        super(Block, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads, device1, device2)
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
    def __init__(self, embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor, device1, device2):
        super(GPT2, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.word_embeddings = nn.Embedding(vocab_size, embed_size).to(self.device1)
        self.position_embeddings = nn.Embedding(max_length, embed_size).to(self.device1)

        # 分割 Transformer 层：前半部分在 device1，后半部分在 device2
        self.layers1 = nn.ModuleList([Block(embed_size, heads, dropout, expansion_factor, device1, device2) for _ in range(num_layers // 2)]).to(self.device1)
        self.layers2 = nn.ModuleList([Block(embed_size, heads, dropout, expansion_factor, device1, device2) for _ in range(num_layers // 2)]).to(self.device2)

        self.fc_out = nn.Linear(embed_size, vocab_size).to(self.device2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)

        out = self.word_embeddings(x) + self.position_embeddings(positions)
        out = self.dropout(out)

        # 第一部分 Transformer 层计算（在 device1）
        for layer in self.layers1:
            out = layer(out)

        # 第二部分 Transformer 层计算（在 device2）
        out = out.to(self.device2)  # 将数据移到 device2
        for layer in self.layers2:
            out = layer(out)

        out = self.fc_out(out)  # 最后的全连接层在 device2 上
        return out

# 初始化分布式训练
def init_distributed_mode(args):
    dist.init_process_group(backend='nccl', init_method="env://")
    torch.cuda.set_device(args.local_rank)

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

# 训练模型
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, help='Local rank for distributed training')
args = parser.parse_args()

init_distributed_mode(args)

device1 = torch.device("cuda:0")
device2 = torch.device("cuda:1")

model = GPT2(embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor, device1, device2).to(device1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
scaler = GradScaler()

batch_time = []
for epoch in range(num_epochs):
    model.train()
    epoch_start_time = time.time()
    for i, batch in enumerate(train_loader):
        batch_start_time = time.time()
        batch = batch.to(device1).long()
        optimizer.zero_grad()
        with autocast():
            output = model(batch)
            loss = loss_fn(output.view(-1, vocab_size), batch.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_end_time = time.time()
        batch_time.append(batch_end_time - batch_start_time)
        print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}, Time: {batch_end_time - batch_start_time:.4f}s")
        if i % 1000 == 0:
            print(f"Avg time per batch: {sum(batch_time) / len(batch_time):.4f}s")
    
    epoch_end_time = time.time()
    print(f"Epoch {epoch + 1} completed in {(epoch_end_time-epoch_start_time):.4f}s")
