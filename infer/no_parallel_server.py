import torch
import torch.nn as nn
from flask import Flask, request, jsonify

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

# 设置超参数
embed_size = 1024
heads = 16
num_layers = 24
vocab_size = 50257
max_length = 1024
dropout = 0.1
expansion_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
def load_model_for_inference(model_path, device):
    model = GPT2(embed_size, heads, num_layers, vocab_size, max_length, dropout, expansion_factor).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded for inference")
    return model

# 加载模型
gpt2_model_path = "../model/gpt2_model.pth"
inference_model = load_model_for_inference(gpt2_model_path, device)

# Flask 服务
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    input_text = data.get("input_text", "")

    # 处理输入文本
    tokenizer = lambda text: [ord(c) for c in text]
    tokens = tokenizer(input_text)[:max_length]
    input_tensor = torch.tensor([tokens]).to(device).long()

    # 模型推理
    with torch.no_grad():
        output = inference_model(input_tensor)
    generated_ids = output.argmax(dim=-1).squeeze(0).tolist()

    # 将生成的 ID 转换为字符
    generated_text = ''.join([chr(id) for id in generated_ids if id != 0])

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
