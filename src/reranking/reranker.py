def compute_score(query: str, document: str) -> float:
    """
    Computes a relevance score between query and document using simple heuristics.
    The score is based on:
    - keyword overlap
    - frequency of important terms
    """

    query_terms = query.lower().split()
    doc_terms = document.lower().split()
    score = 0

    # Count how many query terms appear in the document
    for term in query_terms:
        if term in doc_terms:
            score += 1

    return score


def rerank(query: str, results: list):
    """
    Re-rank retrieved documents based on relevance to the query.
    """

    scored_results = []
    for doc in results:
        score = compute_score(query, doc["text"])
        # Attach score to document
        doc["score"] = score
        scored_results.append(doc)

    # Sort by score descending
    scored_results = sorted(
        scored_results,
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_results
