import numpy as np
from collections import Counter

class Dataset:
    def __init__(self, corpus_file, window_size=2, min_freq=2):
        self.window_size = window_size

        #  Read sentence-based corpus
        with open(corpus_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip().split() for line in f if line.strip()]

        #  Flatten words
        all_words = [w for sent in sentences for w in sent]

        #  Remove rare + junk words
        freq = Counter(all_words)

        # keep only meaningful words
        self.vocab = [w for w in freq if freq[w] >= min_freq and len(w) > 2]

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        #  Build dataset with filtered words
        self.sentences = []
        for sent in sentences:
            filtered = [w for w in sent if w in self.word2idx]
            if len(filtered) > 2:
                self.sentences.append(filtered)

        # convert to indices
        self.data = [[self.word2idx[w] for w in sent] for sent in self.sentences]

        self.vocab_size = len(self.vocab)

        print("\n===== DATASET BUILT =====")
        print(f"Total Sentences: {len(self.sentences)}")
        print(f"Vocabulary Size: {self.vocab_size}")
        print("=========================\n")

    #  CBOW
    def generate_cbow(self):
        X, Y = [], []

        for sent in self.data:
            for i in range(self.window_size, len(sent) - self.window_size):
                context = []
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0:
                        context.append(sent[i + j])

                X.append(context)
                Y.append(sent[i])

        return np.array(X), np.array(Y)

    #  Skip-gram
    def generate_skipgram(self, max_pairs=50000):
        pairs = []

        for sent in self.data:
            for i in range(len(sent)):
                for j in range(-self.window_size, self.window_size + 1):
                    if j != 0 and 0 <= i + j < len(sent):
                        pairs.append((sent[i], sent[i + j]))

                        if len(pairs) >= max_pairs:  # 🔥 limit size
                            return pairs

        return pairs
