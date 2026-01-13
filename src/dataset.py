import json
import torch

MAX_LEN = 25

class MorphDataset(torch.utils.data.Dataset):
    def __init__(self, path, vocab):
        raw = [json.loads(l) for l in open(path, encoding="utf8")]
        self.vocab = vocab

        # Sadece tutarlı olanları al
        self.data = []
        for d in raw:
            if "".join(d["chunks"]) == d["word"] and len(d["word"]) <= MAX_LEN:
                self.data.append(d)

        print("Loaded samples:", len(self.data))

    def encode(self, word):
        x = [self.vocab[c] for c in word]
        return x + [0] * (MAX_LEN - len(x))

    # boundary = CHUNK SONU (çok önemli)
    def make_labels(self, word, chunks):
        y = [0] * len(word)
        idx = 0
        for c in chunks:
            idx += len(c)
            if idx - 1 < len(y):
                y[idx - 1] = 1
        return y + [0] * (MAX_LEN - len(y))

    # padding mask
    def make_mask(self, word):
        return [1] * len(word) + [0] * (MAX_LEN - len(word))

    def __getitem__(self, i):
        d = self.data[i]
        x = self.encode(d["word"])
        y = self.make_labels(d["word"], d["chunks"])
        # m = self.make_mask(d["word"])

        return (
            torch.tensor(x),
            torch.tensor(y).float(),
            # torch.tensor(m).float()
        )

    def __len__(self):
        return len(self.data)
