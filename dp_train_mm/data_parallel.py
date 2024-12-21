import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import time
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from thop import profile  # 导入 thop 库

# 自定义的 GPT-2 网络结构
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum("nqhd,nkhd->nqkh", [queries, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nqkh,nvhd->nqhd", [attention, values]).reshape(N, seq_length, self.embed_size)

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

# 初始化分布式训练
def setup_distributed():
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://192.168.123.134:2024",  # 主节点的 IP 和端口
        world_size=int(os.environ["WORLD_SIZE"]), 
        rank=int(os.environ["RANK"])
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# 清理分布式环境
def cleanup_distributed():
    dist.destroy_process_group()

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

# 数据预处理
def preprocess_data(data):
    tokenizer = lambda text: [ord(c) for c in text]  # 简单的字符级 tokenizer
    return [tokenizer(item['text'])[:max_length] for item in data]

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

if __name__ == "__main__":
    # 设置分布式
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 加载数据集
    dataset = load_dataset("wangrongsheng/ag_news")
    train_tokens = preprocess_data(dataset['train'])
    test_tokens = preprocess_data(dataset['test'])

    # 创建自定义数据集
    train_dataset = CustomDataset(train_tokens)
    test_dataset = CustomDataset(test_tokens)

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn
    )

    # 初始化模型并移动到 GPU
    model = GPT2(embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler()

    # Training loop with batch time tracking
    batch_times = []  # List to store time taken for each batch
    total_flops = 0  # 用于统计所有机器的 FLOPS 总和

    for epoch in range(num_epochs):
        model.train()
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)  # Reset sampler every epoch

        for i, batch in enumerate(train_loader):
            batch_start_time = time.time()  # Start timer for the current batch
            batch = batch.to(device).long()

            # 计算 FLOPS
            flops, _ = profile(model, inputs=(batch,))  # 使用 thop.profile 计算 FLOPS
            total_flops += flops  # 累加 FLOPS

            optimizer.zero_grad()

            with autocast():
                output = model(batch)
                loss = loss_fn(output.view(-1, vocab_size), batch.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            batch_end_time = time.time()  # End timer for the current batch
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)  # Store batch time
            
            if local_rank == 0:
                # 打印每个 batch 的 FLOPS 和处理时间
                print(f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}, Batch Time: {batch_time:.4f}s, FLOPS: {flops / 1e9:.2f} GFLOPS")

        epoch_end_time = time.time()
        if local_rank == 0:
            print(f"Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.4f}s")
            print(f"Total FLOPS for this epoch: {total_flops / 1e9:.2f} GFLOPS")
    
    # 清理分布式环境
    cleanup_distributed()
