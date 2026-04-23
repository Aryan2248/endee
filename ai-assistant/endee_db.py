import numpy as np

class SimpleEndeeDB:
    def __init__(self):
        self.data = []

    def add(self, text, embedding):
        self.data.append({
            "text": text,
            "embedding": embedding
        })

    def search(self, query_embedding, similarity_fn, top_k=1):
        if not self.data:
            return []
            
        scored = []
        for item in self.data:
            score = similarity_fn(query_embedding, item["embedding"])
            scored.append((score, item["text"]))

        # Sort by highest similarity score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]