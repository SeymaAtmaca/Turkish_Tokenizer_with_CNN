import torch
from src.model import BoundaryCNN

MAX_LEN = 25

# modeli ve vocab'ı yükle
state, vocab = torch.load("model.pt", map_location="cpu")

model = BoundaryCNN(len(vocab))
model.load_state_dict(state)
model.eval()

def segment(word):
    x = [vocab.get(c, 0) for c in word]
    x = x + [0] * (MAX_LEN - len(x))
    x = torch.tensor([x])

    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0]

    out = []
    cur = ""

    for i, ch in enumerate(word):
        cur += ch
        if probs[i] > 0.5:
            out.append(cur)
            cur = ""

    if cur:
        out.append(cur)

    return out


# test
print(segment("kızdıklarındakilerdenmiş"))
# print(segment("boyadıklarımızda"))
