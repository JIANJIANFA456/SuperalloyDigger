**Model description**
----------------------
The word embedding model for superalloy corpus was pre-trained on ~9000 unlabeled full-text superalloy articles by Word2Vec continuous bag of words (CBOW) in gensim, which use information about the co-occurrences of words by assigning high-dimensional vectors (embeddings) to words in a text corpus, to preserve their syntactic and semantic relationships.

**Model analysis**
----------------------
Use `analyze_model.py` to inspect the pre-trained model:

```bash
# Print analysis to stdout
python analyze_model.py --model word2vec/all-text

# Save analysis to a file and show top-30 similar words per term
python analyze_model.py --model word2vec/all-text --topn 30 --output analysis.txt

# Also generate a 2-D word-vector scatter plot (requires matplotlib)
python analyze_model.py --model word2vec/all-text --output analysis.txt --visualize
```

The script reports:
- Vocabulary size and vector dimensionality
- Training hyper-parameters (architecture, window size, min-count, epochs)
- Top-N most-similar words for key superalloy property terms (solvus, liquidus, solidus, density) and common chemical elements
- Word analogy test results
- Optional 2-D PCA scatter plot of word vectors saved as a PNG file

**Query similar words**
----------------------
Use `load_word2vec.py` (via `main.py`) to query the most-similar words for any single term:

```bash
python main.py   # uses defaults: model=word2vec/all-text, term=solvus, topn=100
```
