from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_embeddings(model, dataset, num_words=50):
    words = list(dataset.word2idx.keys())[:num_words]
    vectors = [model.W[dataset.word2idx[w]] for w in words]

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(8,6))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x, y, word)

    plt.show()