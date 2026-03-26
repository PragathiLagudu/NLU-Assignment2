from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(corpus_file):
    with open(corpus_file, 'r') as f:
        text = f.read()

    wc = WordCloud(width=800, height=400).generate(text)

    plt.imshow(wc)
    plt.axis("off")
    plt.show()