import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))

""" Generates embeddings for a list of chunked documents. Returns: numpy.ndarray: Embeddings matrix ------------------------- """
def get_embeddings(documents):
    texts = [doc["text"] for doc in documents]   # Extract only the text field from each document
    if not texts:
        raise ValueError("No documents were loaded. Check that the data folder contains readable PDF, DOCX or TXT files.")
    
    return model.encode(
        texts,
        batch_size=32,              # Improves performance.
        show_progress_bar=True,     # Useful for debbuging
        normalize_embeddings=True   # Normalized vector (0-1)
    )
