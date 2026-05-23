import os
import numpy as np
import time
from retrieval.docs_loader import load_documents_from_folder, export_documents_to_word
from retrieval.embeddings import get_embeddings, model
from retrieval.retriever import VectorStore
from retrieval.dictionary import build_synonym_dictionary
from retrieval.query_process import rewrite_and_expand_query
from reranking.reranker import rerank
from orquestacion.prompt_template import build_prompt
from orquestacion.llm import generate_answer

def main():
    
    t0_total = time.perf_counter()

    # Step 1: Load and chunk documents
    base_dir = os.path.dirname(os.path.dirname(__file__)) #  Build a robust absolute path (avoids path errors)
    file_path = os.path.join(base_dir, "data")
    documents = load_documents_from_folder(file_path)
    """ export_documents_to_word(documents, "chunks_output.docx") """

    # Step 2: Generate embeddings
    t0_embeddings = time.perf_counter()
    embeddings = get_embeddings(documents)
    t1_embeddings = time.perf_counter()

    # Step 3: Initialize vector database
    vector_store = VectorStore(documents, embeddings)
    
    # Step 4: user query
    query = input("Pregunta: ")
    
    # Step 5: rewrite and expand query
    synonyms = build_synonym_dictionary(documents)
    expanded_queries = rewrite_and_expand_query(query, synonyms)
    # Step 6: embed all queries
    t0_query_embed = time.perf_counter()
    query_embeddings = model.encode([expanded_queries], normalize_embeddings=True)
    t1_query_embed = time.perf_counter()
    
    # Step 7: search for each query and merge results   
    t0_search = time.perf_counter()
    results = vector_store.search(query_embeddings, k=10)  # Búsqueda semántica en base vectorial (FAISS). Trae los k chunks mas similares.
    t1_search = time.perf_counter()

    # Step 8: Apply re-ranking and show results
    reranked_results = rerank(expanded_queries, results)
    top = 4
    print("-----------------------------------------------------------------------------------")
    print(f"⏱️ Tiempo embeddings documentos: {t1_embeddings - t0_embeddings:.3f} s")
    print(f"Rewritten and expanded queries: {expanded_queries}")
    print(f"⏱️ Tiempo embedding query: {t1_query_embed - t0_query_embed:.6f} s")
    print(f"⏱️ Tiempo búsqueda FAISS: {t1_search - t0_search:.6f} s")
    top_chunks = reranked_results[:top] # Take top chunks after re-ranking
    """ print("\n🔍 Results after query expansion - TOP "+ str(top) +":\n")
    for r in top_chunks:
        print(f"ID: {r['id']}")
        print(f"Text: {r['text']}")
        print("-" * 40) """
        
    t1_total = time.perf_counter()
    print(f"\n⏱️ Tiempo total ejecución: {t1_total - t0_total:.3f} s")
    print("-----------------------------------------------------------------------------------")

    # Step 9: Build prompt
    prompt = build_prompt(query, top_chunks)
   
    # Step 10:Generate answer using LLM
    answer = generate_answer(prompt)
    print("\n✅ Final Answer:\n")
    print(answer)
    print("---------------------------------------------------------------------------------")
   

if __name__ == "__main__":
    main()