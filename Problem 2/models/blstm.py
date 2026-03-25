import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        k = 0.05

        self.W = nn.Parameter(torch.randn(4 * hidden_dim, in_dim) * k)
        self.U = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim) * k)
        self.b = nn.Parameter(torch.zeros(4 * hidden_dim))

    def forward(self, x, h, c):
        gates = x @ self.W.T + h @ self.U.T + self.b

        i, f, o, g = torch.chunk(gates, 4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class BLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.3)

        # forward + backward cells
        self.f_cell = LSTMCell(embed_dim, hidden_dim)
        self.b_cell = LSTMCell(embed_dim, hidden_dim)

        self.output_layer = nn.Linear(2 * hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        device = x.device

        x = self.dropout(self.embedding(x))

        # init states
        hf = torch.zeros(B, self.hidden_dim, device=device)
        cf = torch.zeros(B, self.hidden_dim, device=device)

        hb = torch.zeros(B, self.hidden_dim, device=device)
        cb = torch.zeros(B, self.hidden_dim, device=device)

        # allocate output tensor directly 
        outputs = torch.zeros(B, T, 2 * self.hidden_dim, device=device)

        # forward pass
        for t in range(T):
            hf, cf = self.f_cell(x[:, t], hf, cf)
            outputs[:, t, :self.hidden_dim] = hf

        # backward pass
        for t in reversed(range(T)):
            hb, cb = self.b_cell(x[:, t], hb, cb)
            outputs[:, t, self.hidden_dim:] = hb

        return self.output_layer(outputs)

    @torch.no_grad()
    def generate(self, dataset, max_len=15, temperature=0.6):
        self.eval()

        seq = [dataset.stoi["<"]]

        for _ in range(max_len):
            inp = torch.tensor([seq])

            logits = self.forward(inp)[0, -1]

            logits = logits / temperature
            probs = torch.softmax(logits, dim=0)

            # mix greedy + sampling 
            if torch.rand(1).item() < 0.7:
                idx = torch.argmax(probs).item()
            else:
                idx = torch.multinomial(probs, 1).item()

            if dataset.itos[idx] == ">":
                if len(seq) < 4:
                    continue
                break

            seq.append(idx)

        return "".join(dataset.itos[i] for i in seq[1:])