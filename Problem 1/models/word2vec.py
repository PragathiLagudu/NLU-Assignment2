import numpy as np
from tqdm import tqdm
import random

class Word2Vec:
    def __init__(self, vocab_size, embed_dim=50, neg_samples=5):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.neg_samples = neg_samples

        # input embeddings
        self.W = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # output embeddings
        self.W_out = np.random.randn(vocab_size, embed_dim) * 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 🔥 SKIP-GRAM WITH NEGATIVE SAMPLING
    def train_skipgram(self, pairs, lr=0.01, epochs=5):
        for epoch in range(epochs):
            loss = 0

            for target, context in tqdm(pairs):
                v_w = self.W[target]

                #  positive sample
                score = np.dot(self.W_out[context], v_w)
                pred = self.sigmoid(score)

                loss += -np.log(pred + 1e-9)

                grad = pred - 1

                self.W_out[context] -= lr * grad * v_w
                self.W[target] -= lr * grad * self.W_out[context]

                #  negative samples
                for _ in range(self.neg_samples):
                    neg_word = random.randint(0, self.vocab_size - 1)

                    if neg_word == context:
                        continue

                    score_neg = np.dot(self.W_out[neg_word], v_w)
                    pred_neg = self.sigmoid(score_neg)

                    loss += -np.log(1 - pred_neg + 1e-9)

                    grad_neg = pred_neg

                    self.W_out[neg_word] -= lr * grad_neg * v_w
                    self.W[target] -= lr * grad_neg * self.W_out[neg_word]

            print(f"Skipgram Epoch {epoch+1}, Loss: {loss}")


    #  CBOW WITH NEGATIVE SAMPLING
    def train_cbow(self, X, Y, lr=0.01, epochs=5):
        for epoch in range(epochs):
            loss = 0

            for i in tqdm(range(len(X))):
                context = X[i]
                target = Y[i]

                # average context embeddings
                h = np.mean(self.W[context], axis=0)

                #  positive sample
                score = np.dot(self.W_out[target], h)
                pred = self.sigmoid(score)

                loss += -np.log(pred + 1e-9)

                grad = pred - 1

                self.W_out[target] -= lr * grad * h

                for idx in context:
                    self.W[idx] -= lr * grad * self.W_out[target] / len(context)

                #  negative samples
                for _ in range(self.neg_samples):
                    neg_word = random.randint(0, self.vocab_size - 1)

                    if neg_word == target:
                        continue

                    score_neg = np.dot(self.W_out[neg_word], h)
                    pred_neg = self.sigmoid(score_neg)

                    loss += -np.log(1 - pred_neg + 1e-9)

                    grad_neg = pred_neg

                    self.W_out[neg_word] -= lr * grad_neg * h

                    for idx in context:
                        self.W[idx] -= lr * grad_neg * self.W_out[neg_word] / len(context)

            print(f"CBOW Epoch {epoch+1}, Loss: {loss}")
