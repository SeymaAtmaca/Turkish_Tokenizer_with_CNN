import torch
from src.dataset import MorphDataset
from src.model import BoundaryCNN



state,vocab=torch.load("model.pt")
ivocab={v:k for k,v in vocab.items()}

model=BoundaryCNN(len(vocab))
model.load_state_dict(state)
model.eval()

word="boyadıklar"
x=torch.tensor([[vocab[c] for c in word]+[0]*(25-len(word))])

with torch.no_grad():
    p=model(x)[0][:len(word)]

boundaries=(p>0.5).int()

tokens=[]
start=0
for i,b in enumerate(boundaries):
    if b==1 and i!=0:
        tokens.append(word[start:i])
        start=i
tokens.append(word[start:])

print(tokens)
