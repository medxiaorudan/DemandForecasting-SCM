from typing import List, Dict
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGAssistant:
    def __init__(self, docs_path: str = "docs"):
        self.docs = []
        p = Path(docs_path)
        for md in p.glob("*.md"):
            self.docs.append({"name": md.name, "text": md.read_text(encoding="utf-8")})
        self.vectorizer = TfidfVectorizer(stop_words="english")
        corpus = [d["text"] for d in self.docs]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.docs or self.matrix is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        out = []
        for i in idxs:
            d = self.docs[i]
            snippet = d["text"][:240].replace("\n", " ")
            out.append({"doc": d["name"], "score": float(sims[i]), "snippet": snippet})
        return out
