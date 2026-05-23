import re
import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))

""" Extract most frequent words (simple TF) ---------------------------------------------------------- """
def extract_keywords(text, top_k=10):
    words = re.findall(r'\b\w+\b', text.lower())

    # ✅ remove short words AND pure numbers
    words = [
        w for w in words
        if len(w) > 3 and not w.isdigit()
    ]

    freq = Counter(words)
    keywords = [w for w, _ in freq.most_common(top_k)]

    return list(set(keywords))



""" Groups similar words based on cosine similarity -------------------------------------------------"""
def group_synonyms(words, threshold=0.75):
    if not words:
        return []

    embeddings = model.encode(words, normalize_embeddings=True)
    groups = []
    used = set()

    for i, word in enumerate(words):
        if word in used:
            continue
        group = [word]
        used.add(word)
        for j in range(i + 1, len(words)):
            if words[j] in used:
                continue
            # cosine similarity
            sim = np.dot(embeddings[i], embeddings[j])
            if sim > threshold:
                group.append(words[j])
                used.add(words[j])
        groups.append(group)

    return groups


""" Builds a synonym dictionary incrementally from all documents -------------------------------------------------------"""
def build_synonym_dictionary(documents):

    # Keywords
    all_keywords = []

    for doc in documents:
        keywords = extract_keywords(doc["text"])
        all_keywords.extend(keywords)

    # Duplicated
    all_keywords = list(set(all_keywords))

    # Group synonyms
    synonym_groups = group_synonyms(all_keywords)

    synonym_dict = {}
    for group in synonym_groups:
        for word in group:
            synonym_dict[word] = [item for item in group if item != word]

    return synonym_dict
