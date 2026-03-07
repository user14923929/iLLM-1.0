import torch
import torch.nn.functional as F

def generate(model, context, length=100):

    for _ in range(length):

        logits = model(context)

        probs = F.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat((context, next_token), dim=1)

    return context