# NLU-Assignment 2
## Name: Lagudu Sai Pragathi Roll No: B23CM1021
# Problem 1
# Word2Vec on IIT Jodhpur Data 

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
Problem 1/
│
├── data/
│   ├── raw/                  # Raw text files (PDF extracted )
│   └── corpus.txt            # Preprocessed corpus
│
├── utils/
│   ├── preprocess.py         # Text cleaning & corpus creation
│   └── dataset.py            # Dataset & training pair generation
│   
│
├── models/
│   └── word2vec.py           # CBOW & Skip-gram (Negative Sampling)
│
├── analysis/
│   ├── evaluate.py           # Nearest neighbors & analogy
│   ├── visualize.py          #  t-SNE plots
│   └── wordcloud_plot.py     # Word cloud generation
│
├── main.py
├──requirements.txt                # End-to-end pipeline
└── README.md
```

---

## Dataset Preparation

### Sources Used

* Academic regulation documents 
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
# Problem 2
# Character-Level Name Generation using RNN, BLSTM, and Attention

## Overview
This project implements and compares three sequence models for generating Indian names at the character level:
- Vanilla RNN
- Bidirectional LSTM (BLSTM)
- Attention-based RNN

---

## Dataset
- 1000 Indian names generated using LLM
- Stored in `TrainingNames.txt`
- Each name is processed character-by-character

---
## Project Struture
```
Problem 2/
│
├── main.py                     # Runs all models (train + generate + evaluate)
├── TrainingNames.txt          # Dataset (1000 Indian names)
├── README.md                  # Project documentation
│
├── models/                    # All model implementations
│   ├── rnn.py                 # Vanilla RNN (with generate function)
│   ├── blstm.py               # BLSTM (manual implementation + generate)
│   └── attention_rnn.py       # Attention RNN (with generate)
│
├── utils/                     # Utility functions
│   ├── dataset.py             # Data loading, encoding, vocabulary
│   ├── train.py               # Training loop
│   └── evaluate.py            # Novelty & diversity metrics
│
├── outputs/                   # Generated results
│   ├── rnn_samples.txt
│   ├── blstm_samples.txt
│   └── attention_samples.txt
```

## Models Implemented

###  Vanilla RNN
- 2-layer RNN with embedding and dropout
- Produces diverse outputs

###  BLSTM
- Bidirectional LSTM (manual implementation)
- Captures forward and backward context

###  Attention RNN
- RNN with dot-product attention
- Focuses on important character patterns

---

##  How to Run

```bash
python main.py
```
---
## Results
```
Model  	           Novelty	  Diversity
Vanilla RNN	        0.915	     0.92
BLSTM	             1.0       	1.0
Attention RNN      	0.905	     0.70
```

## Sample Outputs
```
RNN: aarita, manda, sarina
BLSTM: zo, es
Attention: aarati, shruti, manita
```

## Key Insights
Attention RNN produced the most realistic names
Vanilla RNN had highest diversity but some noise
BLSTM struggled with generation due to bidirectional dependency

## Technologies Used
Python
PyTorch
NumPy
