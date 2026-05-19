import numpy as np
import pandas as pd
import math


class TFIDFVectorizer:
    """TF-IDF Vectorizer implemented completely from scratch."""

    def __init__(self, max_features=3000):
        self.max_features = max_features
        self.vocab = {}          # word -> index
        self.idf_values = {}     # word -> idf score

    def _tokenize(self, text):
        return str(text).lower().split()

    def fit(self, texts):
        # Count document frequency for each word
        df_counts = {}
        N = len(texts)

        for text in texts:
            tokens = set(self._tokenize(text))
            for token in tokens:
                df_counts[token] = df_counts.get(token, 0) + 1

        # Compute IDF: log(N / df) + 1
        idf_all = {}
        for word, df in df_counts.items():
            idf_all[word] = math.log(N / df) + 1

        # Keep top max_features words by IDF (most informative)
        sorted_words = sorted(idf_all.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_words[:self.max_features]

        self.vocab = {word: idx for idx, (word, _) in enumerate(selected)}
        self.idf_values = {word: idf for word, idf in selected}
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            tokens = self._tokenize(text)
            total = len(tokens) if len(tokens) > 0 else 1

            # Compute TF
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in tf:
                tf[token] = tf[token] / total

            # Build TF-IDF vector
            vec = np.zeros(len(self.vocab))
            for token, tf_val in tf.items():
                if token in self.vocab:
                    idx = self.vocab[token]
                    vec[idx] = tf_val * self.idf_values[token]

            rows.append(vec)
        return np.array(rows)

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)
