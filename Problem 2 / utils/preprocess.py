import re
import os
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def read_file_safely(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(path, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            with open(path, 'r', errors='ignore') as f:
                return f.read()

def clean_text(text):
    text = text.lower()

    # remove html tags
    text = re.sub(r'<.*?>', ' ', text)

    # remove urls
    text = re.sub(r'http\S+', ' ', text)

    # keep only alphabets
    text = re.sub(r'[^a-z\s]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    text = text.replace('\xa0', ' ').replace('\ufeff', ' ')

    words = text.split()

    # remove stopwords + short words
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    return words

def build_corpus(input_folder, output_file):
    all_words = []
    seen_sentences = set()

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)

        text = read_file_safely(path)

        words = clean_text(text)
        sentence = " ".join(words)

        if sentence not in seen_sentences:
            seen_sentences.add(sentence)
            all_words.extend(words)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(" ".join(all_words))

    print(f"Saved clean corpus with {len(all_words)} words")