import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.3)

        self.rnn = nn.RNN(
            embed_size,
            hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)

        out, _ = self.rnn(x)
        out = self.fc(out)

        return out

    @torch.no_grad()
    def generate(self, dataset, max_len=15, temperature=0.8, top_k=5):
        self.eval()

        seq = [dataset.stoi["<"]]

        for _ in range(max_len):
            x = torch.tensor([seq])

            logits = self.forward(x)[0, -1]

            # temperature scaling
            logits = logits / temperature

            # top-k filtering
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                filtered = torch.full_like(logits, -1e9)
                filtered[indices] = values
                logits = filtered

            probs = torch.softmax(logits, dim=0)

            # sampling
            idx = torch.multinomial(probs, 1).item()

            # EOS handling
            if dataset.itos[idx] == ">":
                if len(seq) < 3:
                    continue
                break

            seq.append(idx)

        return "".join(dataset.itos[i] for i in seq[1:])