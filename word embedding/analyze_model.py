# -*- coding: utf-8 -*-
"""
Analyze the pre-trained Word2Vec model for superalloy corpus.

Usage:
    python analyze_model.py --model word2vec/all-text [--topn 20] [--output analysis.txt] [--visualize]

This script reports:
  - Vocabulary size and vector dimensionality
  - Top-N most similar words for each key superalloy property term
  - Word analogy test results
  - Optional 2-D visualization of selected word vectors (PCA)
"""

import argparse
import os
import sys

from gensim.models import word2vec


# Key superalloy property terms to analyze
PROPERTY_TERMS = ["solvus", "liquidus", "solidus", "density"]

# Chemical elements commonly found in superalloys
ELEMENT_TERMS = ["Ni", "Co", "Cr", "Al", "Ti", "W", "Mo", "Re", "Ta", "Nb"]

# Analogy queries in the form (a, b, c) → d  where  a - b + c ≈ d
ANALOGY_TESTS = [
    ("solvus", "temperature", "density", "measurement"),
    ("Ni", "nickel", "Co", "cobalt"),
]


def load_model(model_path):
    """Load a gensim Word2Vec model from *model_path*.

    Returns the loaded model, or raises FileNotFoundError if the path does not
    exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found: {}\n"
            "Please provide the path to a pre-trained Word2Vec model.".format(model_path)
        )
    model = word2vec.Word2Vec.load(model_path)
    return model


def report_basic_info(model, out):
    """Write basic model statistics to *out*."""
    vocab_size = len(model.wv)
    vector_size = model.wv.vector_size
    out.write("=" * 60 + "\n")
    out.write("Word2Vec Model Analysis — SuperalloyDigger\n")
    out.write("=" * 60 + "\n\n")
    out.write("Vocabulary size : {:,}\n".format(vocab_size))
    out.write("Vector size     : {}\n".format(vector_size))

    # Report training hyper-parameters when available
    if hasattr(model, "epochs"):
        out.write("Training epochs : {}\n".format(model.epochs))
    if hasattr(model, "window"):
        out.write("Window size     : {}\n".format(model.window))
    if hasattr(model, "min_count"):
        out.write("Min word count  : {}\n".format(model.min_count))
    if hasattr(model, "sg"):
        arch = "Skip-gram" if model.sg else "CBOW"
        out.write("Architecture    : {}\n".format(arch))
    out.write("\n")


def report_similar_words(model, terms, topn, out):
    """Write the *topn* most-similar words for each term in *terms* to *out*."""
    out.write("-" * 60 + "\n")
    out.write("Most-similar words (top {})\n".format(topn))
    out.write("-" * 60 + "\n\n")
    for term in terms:
        if term not in model.wv:
            out.write("  [{}]  — not in vocabulary\n\n".format(term))
            continue
        similar = model.wv.most_similar(term, topn=topn)
        out.write("  [{}]\n".format(term))
        for word, score in similar:
            out.write("    {:<30s}  {:.4f}\n".format(word, score))
        out.write("\n")


def report_analogies(model, tests, out):
    """Write word-analogy test results to *out*.

    For each (a, b, c, expected) quadruple, queries  a - b + c  and reports
    whether *expected* appears in the top results.
    """
    out.write("-" * 60 + "\n")
    out.write("Word analogy tests  (a – b + c → ?)\n")
    out.write("-" * 60 + "\n\n")
    for a, b, c, expected in tests:
        missing = [t for t in (a, b, c) if t not in model.wv]
        if missing:
            out.write(
                "  {} – {} + {}  →  skipped (missing: {})\n\n".format(
                    a, b, c, ", ".join(missing)
                )
            )
            continue
        results = model.wv.most_similar(positive=[a, c], negative=[b], topn=5)
        top_words = [w for w, _ in results]
        hit = "✓" if expected in top_words else "✗"
        out.write("  {} – {} + {}  →  {}\n".format(a, b, c, ", ".join(top_words)))
        out.write("    Expected '{}': {}\n\n".format(expected, hit))


def visualize_vectors(model, terms, output_path):
    """Generate a 2-D scatter plot of word vectors for *terms* using PCA.

    Falls back to projecting onto the first two raw dimensions when
    scikit-learn is not available.  Saves the figure to *output_path*.
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib / numpy not available; skipping visualization.")
        return

    in_vocab = [t for t in terms if t in model.wv]
    if len(in_vocab) < 2:
        print("Not enough terms in vocabulary for visualization; skipping.")
        return

    vectors = np.array([model.wv[t] for t in in_vocab])

    # Try PCA first
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors)
        method_label = "PCA"
    except ImportError:
        # Fall back to a simple 2-component projection
        coords = vectors[:, :2]
        method_label = "first 2 dims"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    for i, word in enumerate(in_vocab):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=9)
    ax.set_title("Word vector projection ({}) — SuperalloyDigger".format(method_label))
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print("Visualization saved to: {}".format(output_path))


def analyze(model_path, topn=20, output_path=None, visualize=False):
    """Run the full model analysis and write results to *output_path* (or stdout)."""
    model = load_model(model_path)

    if output_path:
        out_file = open(output_path, "w", encoding="utf-8")
    else:
        out_file = sys.stdout

    try:
        report_basic_info(model, out_file)
        report_similar_words(model, PROPERTY_TERMS + ELEMENT_TERMS, topn, out_file)
        report_analogies(model, ANALOGY_TESTS, out_file)
    finally:
        if output_path:
            out_file.close()

    if output_path:
        print("Analysis written to: {}".format(output_path))

    if visualize:
        vis_path = (
            os.path.splitext(output_path)[0] + "_vectors.png"
            if output_path
            else "model_vectors.png"
        )
        visualize_vectors(model, PROPERTY_TERMS + ELEMENT_TERMS, vis_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the pre-trained Word2Vec model for superalloy corpus."
    )
    parser.add_argument(
        "--model",
        default="word2vec/all-text",
        help="Path to the pre-trained Word2Vec model (default: word2vec/all-text)",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=20,
        help="Number of most-similar words to report per term (default: 20)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the text analysis output file (default: stdout)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate a 2-D scatter plot of word vectors and save as a PNG file",
    )
    args = parser.parse_args()
    analyze(args.model, topn=args.topn, output_path=args.output, visualize=args.visualize)


if __name__ == "__main__":
    main()
