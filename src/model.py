import torch
import torch.nn as nn

class BoundaryCNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.emb=nn.Embedding(vocab_size+1,64,padding_idx=0)

        self.net=nn.Sequential(
            nn.Conv1d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv1d(128,256,5,padding=2),
            nn.ReLU(),
            nn.Conv1d(256,256,7,padding=3),
            nn.ReLU(),
            nn.Conv1d(256,1,1)
        )

    def forward(self,x):
        x=self.emb(x).permute(0,2,1)
        x=self.net(x).squeeze(1)
        return x   # ❗ sigmoid YOK
