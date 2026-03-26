import re
import os
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')


IMPORTANT_WORDS = {
    "ug", "pg", "btech", "mtech", "phd", "research",
    "student", "course", "exam", "academic", "program"
}


def read_file_safely(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            with open(path, 'r', errors='ignore') as f:
                return f.read()


def clean_sentence(sentence):
    sentence = sentence.lower()

    # remove html
    sentence = re.sub(r'<.*?>', ' ', sentence)

    # remove urls
    sentence = re.sub(r'http\S+', ' ', sentence)

    # fix pdf garbage
    sentence = sentence.replace('\xa0', ' ').replace('\ufeff', ' ')

    # keep alphabets + numbers (IMPORTANT)
    sentence = re.sub(r'[^a-z0-9\s]', ' ', sentence)

    # remove very short tokens BUT keep important ones
    words = sentence.split()
    words = [w for w in words if len(w) > 2 or w in IMPORTANT_WORDS]

    return words


def build_corpus(input_folder, output_file):
    corpus_sentences = []
    seen_sentences = set()

    total_docs = 0

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)

        text = read_file_safely(path)
        total_docs += 1

        #  sentence tokenization 
        sentences = sent_tokenize(text)

        for sent in sentences:
            words = clean_sentence(sent)

            if len(words) < 3:
                continue

            clean_sent = " ".join(words)

            # remove duplicate sentences 
            if clean_sent not in seen_sentences:
                seen_sentences.add(clean_sent)
                corpus_sentences.append(clean_sent)

    #  LIMIT SIZE 
    corpus_sentences = corpus_sentences[:5000]

    # save corpus (one sentence per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sent in corpus_sentences:
            f.write(sent + "\n")

    print("\n===== CORPUS BUILT =====")
    print(f"Documents: {total_docs}")
    print(f"Sentences: {len(corpus_sentences)}")
    print("========================\n")

    