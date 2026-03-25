import numpy as np

class Dataset:
    def __init__(self, corpus_file, window_size=2):
        with open(corpus_file, 'r') as f:
            words = f.read().split()

        self.vocab = list(set(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        self.data = [self.word2idx[w] for w in words]

        self.window_size = window_size
        self.vocab_size = len(self.vocab)

    def generate_cbow(self):
        X, Y = [], []

        for i in range(self.window_size, len(self.data) - self.window_size):
            context = []
            for j in range(-self.window_size, self.window_size + 1):
                if j != 0:
                    context.append(self.data[i + j])

            X.append(context)
            Y.append(self.data[i])

        return np.array(X), np.array(Y)

    def generate_skipgram(self):
        pairs = []

        for i in range(self.window_size, len(self.data) - self.window_size):
            for j in range(-self.window_size, self.window_size + 1):
                if j != 0:
                    pairs.append((self.data[i], self.data[i + j]))

        return pairs