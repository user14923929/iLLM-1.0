def build_tokenizer(text):

    chars = sorted(list(set(text)))

    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}

    return stoi, itos


def encode(text, stoi):

    return [stoi[c] for c in text]


def decode(tokens, itos):

    return "".join([itos[i] for i in tokens])