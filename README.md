# NLU-Assignment 2
# Word2Vec on IIT Jodhpur Data (Problem 1)

## Overview

This project focuses on learning **word embeddings** using Word2Vec models trained on textual data collected from IIT Jodhpur sources. The goal is to capture semantic relationships between words and analyze how well the learned embeddings represent academic and research-related concepts.

---

## Objectives

* Build a clean corpus from IIT Jodhpur data sources
* Implement **Word2Vec from scratch**
* Train **CBOW** and **Skip-gram with Negative Sampling** models
* Perform semantic analysis using:

  * Nearest neighbors
  * Word analogies
* Visualize embeddings using **t-SNE**

---

## 📁 Project Structure

```
word2vec_iitj/
│
├── data/
│   ├── raw/                  # Raw text files (PDF extracted / manual text)
│   └── corpus.txt            # Preprocessed corpus
│
├── utils/
│   ├── preprocess.py         # Text cleaning & corpus creation
│   ├── dataset.py            # Dataset & training pair generation
│   └── pdf_extractor.py      # Extract text from PDFs
│
├── models/
│   └── word2vec.py           # CBOW & Skip-gram (Negative Sampling)
│
├── analysis/
│   ├── evaluate.py           # Nearest neighbors & analogy
│   ├── visualize.py          # PCA / t-SNE plots
│   └── wordcloud_plot.py     # Word cloud generation
│
├── main.py                   # End-to-end pipeline
└── README.md
```

---

## Dataset Preparation

### Sources Used

* Academic regulation documents (mandatory)
* Course syllabus
* Research and academic content

### Preprocessing Steps

* Removal of PDF artifacts and boilerplate text
* Lowercasing
* Sentence tokenization
* Removal of punctuation and non-text elements
* Filtering of noisy and rare words

### Dataset Statistics

* Documents: **8**
* Tokens: **~39,000**
* Vocabulary Size: **~4,600**

---

##  Model Implementation

Two models were implemented **from scratch**:

### 1. CBOW (Continuous Bag of Words)

* Predicts target word from context
* Faster and smoother embeddings

### 2. Skip-gram with Negative Sampling

* Predicts context from target word
* Produces better semantic representations

###  Parameters Used

* Embedding dimension: 50 
* Window size: 2 
* Negative samples: 5
* Epochs: 5

---
##  Word Cloud

The word cloud below shows the most frequent words in the corpus:

![Word Cloud](NLU-Assignment2/Problem 1/Output/wc.png)
##  Semantic Analysis

### Nearest Neighbors

Words such as **research, student, phd, exam** were analyzed using cosine similarity.

* CBOW → captures general context
* Skip-gram → captures domain-specific meaning

### Analogy Tasks

Examples:

* `btech : undergraduate :: mtech : postgraduate`
* `phd : research :: mtech : project`
* `student : course :: phd : research`

Results showed **partial semantic correctness**, limited by dataset size and word frequency.

---

##  Visualization

Word embeddings were visualized using:

* **PCA**
* **t-SNE**

### Observations

* Academic terms form clusters
* Program-related words group together
* Skip-gram shows clearer separation than CBOW

---

##  Limitations

* Limited dataset size (~2000 sentences)
* Some important words missing or low frequency
* Presence of frequent/common words affecting results

---

##  Conclusion

* Word2Vec successfully captured semantic relationships in academic text
* Skip-gram performed better than CBOW in semantic tasks
* Data quality and size significantly impact embedding performance

---

##  How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py
```

---

##  Key Insight

> The effectiveness of word embeddings depends more on **data quality and co-occurrence patterns** than model implementation.

---

