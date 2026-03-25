import torch

SOS = "<"
EOS = ">"

class NameDataset:
    def __init__(self, file):
        with open(file) as f:
            self.names = [line.strip().lower() for line in f if line.strip()]

        chars = set("".join(self.names))
        chars.update([SOS, EOS])

        self.chars = sorted(list(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.vocab_size = len(self.chars)

    def encode(self, name):
        return [self.stoi[SOS]] + [self.stoi[c] for c in name] + [self.stoi[EOS]]

    def decode(self, indices):
        return "".join([self.itos[i] for i in indices])
