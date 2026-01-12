import torch, json
from src.dataset import MorphDataset
from src.model import BoundaryCNN
import matplotlib.pyplot as plt


data=[json.loads(l) for l in open("data/train.jsonl",encoding="utf8")]
chars=set("".join(d["word"] for d in data))
vocab={c:i+1 for i,c in enumerate(chars)}

ds=MorphDataset("data/train.jsonl",vocab)
dl=torch.utils.data.DataLoader(ds,batch_size=8,shuffle=True)

model=BoundaryCNN(len(vocab))
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn=torch.nn.BCELoss()
losses=[]

for epoch in range(80):
    for x,y in dl:
        p=model(x)
        loss=loss_fn(p,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(epoch,loss.item())


plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training loss")
plt.grid(True)
plt.show()

torch.save((model.state_dict(),vocab),"model.pt")
