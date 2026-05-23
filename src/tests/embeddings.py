import os
from sentence_transformers import SentenceTransformer
from retrieval.docs_loader import load_documents_from_folder


test_queries = [
    {"query": "Control celda IM SM6-36", "expected": "IC-CC-9.3.20..."},
    {"query": "Como gestiono reclamo de cliente", "expected": "..."},
]


def test_models(models, documents):

    results = {}

    """ for model_name in models:
        print(f"Testing {model_name}")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            [doc["text"] for doc in documents],
            normalize_embeddings=True
        )
        index = build_faiss(embeddings)

        score = evaluate_model(model_name, documents, test_queries, index)

        results[model_name] = score """

    return results



base_dir = os.path.dirname(os.path.dirname(__file__)) #  Build a robust absolute path (avoids path errors)
file_path = os.path.join(base_dir, "data")
documents = load_documents_from_folder(file_path)

models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-base-en-v1.5"
]

results = test_models(models, documents)

print(results)