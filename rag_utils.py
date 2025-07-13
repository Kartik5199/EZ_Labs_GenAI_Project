from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class RAGVectorStore:
    def __init__(self):
        self.documents: List[str] = []
        self.embeddings = None
        self.index = None

    def build_index(self, chunks: List[str]):
        """Embed and index the provided chunks"""
        self.documents = chunks
        self.embeddings = embedder.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
        dim = self.embeddings[0].shape[0]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(self.embeddings))

    def query(self, question: str, top_k: int = 100) -> List[str]:
        """Find top-k most relevant chunks to the question"""
        query_vec = embedder.encode([question], normalize_embeddings=True)
        scores, indices = self.index.search(np.array(query_vec), top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks of words"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks
