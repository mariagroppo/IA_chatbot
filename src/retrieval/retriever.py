import faiss

class VectorStore:
    
    """Basic vector database using FAISS for similarity search. Stores both embeddings and original documents."""
    def __init__(self, documents, embeddings):
        """ Initializes the vector store."""
        self.documents = documents                           # Store documents for later retrieval
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])  # Create FAISS index (L2 distance)
        self.index.add(embeddings)                           # Add embeddings to the index
    
    def search(self, query_embedding, k=3):
        """ Searches for the top-k most similar documents. """
        distances, indices = self.index.search(query_embedding, k)   # Perform similarity search
        results = []
        
        for i in indices[0]:                                         # Retrieve corresponding documents
            results.append(self.documents[i])
        
        return results

