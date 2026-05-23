import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load a lightweight and efficient embedding model
load_dotenv()
model_key= os.getenv("MODEL")
model = SentenceTransformer(os.getenv(model_key))

""" Generates embeddings for a list of chunked documents. Returns: numpy.ndarray: Embeddings matrix ------------------------- """
def get_embeddings(documents):
    texts = [doc["text"] for doc in documents]   # Extract only the text field from each document
    
    embeddings = model.encode(
        texts,
        batch_size=32,              # Improves performance.
        show_progress_bar=True,     # Useful for debbuging
        normalize_embeddings=True   # Normalized vector (0-1)
    )

    return model.encode(texts)                   # Convert text into vector embeddings