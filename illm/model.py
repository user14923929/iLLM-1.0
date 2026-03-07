import torch
import torch.nn as nn

class iLLM(nn.Module):

    def __init__(self, vocab_size, embed_size=256):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        x = self.embed(x)
        x = self.linear(x)

        return x