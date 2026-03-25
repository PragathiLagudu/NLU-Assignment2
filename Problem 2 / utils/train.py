import torch
import torch.nn as nn
import torch.optim as optim

def train(model, dataset, epochs=10, lr=0.002):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(epochs):
        total_loss = 0

        for name in dataset.names:
            seq = dataset.encode(name)

            x = torch.tensor(seq[:-1]).unsqueeze(0)
            y = torch.tensor(seq[1:]).unsqueeze(0)

            optimizer.zero_grad()

            output = model(x)
            loss = loss_fn(output.view(-1, dataset.vocab_size), y.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")