import numpy as np

class Matcher:
    def __init__(self, db):
        self.db = db

    def cosine_similarity(self, a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(b, a)

    def search(self, emb, top_k=1):
        names, embs = self.db.get_all_embeddings()
        if len(embs) == 0:
            return []
        sims = self.cosine_similarity(emb, embs)
        topk_idx = np.argsort(sims)[::-1][:top_k]
        results = [{"name": names[i], "similarity": float(sims[i])} for i in topk_idx]
        return results
