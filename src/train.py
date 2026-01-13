import torch, json
from src.dataset import MorphDataset
from src.model import BoundaryCNN
import matplotlib.pyplot as plt

data=[json.loads(l) for l in open("data/train.jsonl",encoding="utf8")]
chars=set("".join(d["word"] for d in data))
vocab={c:i+1 for i,c in enumerate(chars)}

ds=MorphDataset("data/train.jsonl",vocab)
dl=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = BoundaryCNN(len(vocab)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

pos_weight = torch.tensor([10.0], device=device)
criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

losses = []

for epoch in range(20):
    total = 0

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        mask = (x != 0).float()

        logits = model(x)

        loss = criterion(logits, y)
        loss = (loss * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    avg = total / len(dl)
    losses.append(avg)
    print(f"Epoch {epoch+1}: {avg:.4f}")


plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

torch.save((model.state_dict(),vocab),"model.pt")
