# iLLM-1.0.py
# Fully functional iLLM-1.0: model, training, generation

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =======================
# 1️⃣ Load dataset
# =======================
dataset_file = "iLLM-1.0_dataset.txt"
with open(dataset_file, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i in range(vocab_size)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# =======================
# 2️⃣ Model parameters
# =======================
BLOCK_SIZE = 128
BATCH_SIZE = 64
EMBED_SIZE = 512
N_LAYERS = 8
N_HEADS = 8

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x, y

# =======================
# 3️⃣ Multi-Head Attention & Transformer
# =======================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, T, C = x.shape
        H = self.heads
        values = self.values(x).view(N, T, H, self.head_dim)
        keys = self.keys(x).view(N, T, H, self.head_dim)
        queries = self.queries(x).view(N, T, H, self.head_dim)
        energy = torch.einsum("nthe,nshe->nths", queries, keys) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nths,nshe->nthe", attention, values).reshape(N, T, C)
        return self.fc_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x

class iLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)
        self.layers = nn.ModuleList([TransformerBlock(EMBED_SIZE, N_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_SIZE)
        self.head = nn.Linear(EMBED_SIZE, vocab_size)

    def forward(self, idx):
        N, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T))
        x = tok + pos
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)

# =======================
# 4️⃣ Initialize model
# =======================
model = iLLM()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# =======================
# 5️⃣ Training
# =======================
print("Starting training iLLM-1.0...")
for step in range(2000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step} Loss: {loss.item()}")

# =======================
# 6️⃣ Text generation
# =======================
context = torch.zeros((1,1), dtype=torch.long)
for _ in range(200):
    logits = model(context)
    logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, next_token), dim=1)

print("Generated text:\n", decode(context[0].tolist()))

# =======================
# 7️⃣ Save model
# =======================
torch.save(model.state_dict(), "iLLM-1.0.pt")
print("iLLM-1.0 trained and saved!")