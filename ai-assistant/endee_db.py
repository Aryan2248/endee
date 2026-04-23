import numpy as np

class SimpleEndeeDB:
    def __init__(self):
        self.texts = []
        self.embeddings = None  # numpy matrix (N x D)

    @property
    def data(self):
        # Keep compatibility with app.py's len(st.session_state.db.data)
        return self.texts

    def add(self, text, embedding):
        self.texts.append(text)
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)

        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def search(self, query_embedding, similarity_fn=None, top_k=2):
        if self.embeddings is None or len(self.texts) == 0:
            return []

        # Vectorized cosine similarity — computes ALL scores in one shot
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        normed = self.embeddings / norms

        scores = normed @ q_norm  # dot product across all vectors at once

        # Get top_k indices without full sort (much faster for large DBs)
        top_k = min(top_k, len(self.texts))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [self.texts[i] for i in top_indices]