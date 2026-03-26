import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def nearest_neighbors(model, word, dataset, k=5):
    idx = dataset.word2idx[word]
    vec = model.W[idx]

    sims = []
    for i in range(dataset.vocab_size):
        sim = cosine_similarity(vec, model.W[i])
        sims.append((dataset.idx2word[i], sim))

    sims = sorted(sims, key=lambda x: -x[1])
    return sims[1:k+1]


def analogy(model, dataset, w1, w2, w3):
    for w in [w1, w2, w3]:
        if w not in dataset.word2idx:
            print(f"Word '{w}' not in vocabulary!")
            return None

    v = (
        model.W[dataset.word2idx[w2]]
        - model.W[dataset.word2idx[w1]]
        + model.W[dataset.word2idx[w3]]
    )

    best_word = None
    best_sim = -1

    for i in range(dataset.vocab_size):
        word = dataset.idx2word[i]

        if word in [w1, w2, w3]:
            continue

        sim = cosine_similarity(v, model.W[i])

        if sim > best_sim:
            best_sim = sim
            best_word = word

    return best_word