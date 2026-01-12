import json
import torch


MAX_LEN = 25

class MorphDataset(torch.utils.data.Dataset):
    def __init__(self, path, vocab):
        raw = [json.loads(l) for l in open(path, encoding="utf8")]
        self.vocab = vocab

        # ❗ Yalnızca string olarak tutarlı olanları al
        self.data = []
        for d in raw:
            if "".join(d["chunks"]) == d["word"]:
                self.data.append(d)

        print("Loaded samples:", len(self.data))

    def encode(self, word):
        x = [self.vocab[c] for c in word]
        return x + [0] * (MAX_LEN - len(x))

    def make_labels(self, word, chunks):
        y = [0] * len(word)
        idx = 0
        for c in chunks:
            if idx < len(y):
                y[idx] = 1
            idx += len(c)
        return y + [0] * (MAX_LEN - len(y))

    def __getitem__(self, i):
        d = self.data[i]
        x = self.encode(d["word"])
        y = self.make_labels(d["word"], d["chunks"])
        return torch.tensor(x), torch.tensor(y).float()

    def __len__(self):
        return len(self.data)
