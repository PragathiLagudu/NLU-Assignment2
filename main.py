from utils.preprocess import build_corpus
from utils.dataset import Dataset
from models.word2vec import Word2Vec
from analysis.evaluate import nearest_neighbors, analogy
from analysis.wordcloud import generate_wordcloud
from analysis.visualize import plot_embeddings
from utils.pdf_extractor import extract_pdf_text

# STEP 1: Build corpus
build_corpus("data/raw-text", "data/corpus.txt")

with open("data/clean_corpus.txt", encoding="utf-8") as f:
    words = f.read().split()
num_tokens = len(words)
vocab_size = len(set(words))

print("\n===== DATASET STATS =====")
print(f"Total Tokens: {num_tokens}")
print(f"Vocabulary Size: {vocab_size}")
print("=========================\n")

# STEP 2: Dataset
dataset = Dataset("data/clean_corpus.txt", window_size=2)

# STEP 3: Train CBOW
X, Y = dataset.generate_cbow()

model_cbow = Word2Vec(dataset.vocab_size, embed_dim=50)
model_cbow.train_cbow(X, Y, epochs=5)

# STEP 4: Train Skipgram
pairs = dataset.generate_skipgram()

model_sg = Word2Vec(dataset.vocab_size, embed_dim=50)
model_sg.train_skipgram(pairs, epochs=5)

# STEP 5: Neighbors
words = ["research", "student", "phd", "exam"]

for w in words:
    print(f"\nNearest for by cbow {w}:")
    print(nearest_neighbors(model_cbow, w, dataset))
    print(f"\nNearest for by skipgram {w}:")
    print(nearest_neighbors(model_sg, w, dataset))


# STEP 6: Analogy
print("\ncbow-Analogy (ug : btech :: pg : ?)")
print(analogy(model_cbow, dataset, "ug", "btech", "pg"))
print("\nskipgram-Analogy (ug : btech :: pg : ?)")
print(analogy(model_sg, dataset, "ug", "btech", "pg"))

print("\ncbow-Analogy(student : exam :: researcher : ?)")
print(analogy(model_cbow, dataset, "student", "exam", "researcher"))
print("\nskipgram-Analogy(student : exam :: researcher : ?)")
print(analogy(model_sg, dataset, "student", "exam", "researcher"))

print("\ncbow-Analogy(btech : undergraduate :: phd : ?)")
print(analogy(model_cbow, dataset, "btech", "undergraduate", "phd"))
print("\nskipgram-Analogy(btech : undergraduate :: phd : ?)")
print(analogy(model_sg, dataset, "btech", "undergraduate", "phd"))

# STEP 7: WordCloud
generate_wordcloud("data/clean_corpus.txt")

# STEP 8: Visualization
plot_embeddings(model_cbow, dataset)
plot_embeddings(model_sg, dataset)