import torch
from illm.model import iLLM
from illm.tokenizer import build_tokenizer, encode

with open("iLLM-1.0_dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

stoi, itos = build_tokenizer(text)

data = torch.tensor(encode(text, stoi))

vocab_size = len(stoi)

model = iLLM(vocab_size)

print("Model initialized!")