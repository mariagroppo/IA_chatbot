""" Builds a structured prompt for the LLM using retrieved context ---------------------------------------------------"""
def build_prompt(question: str, context_chunks: list) -> str:

    # Combine top chunks into a single context
    context_text = "\n\n".join([doc["text"] for doc in context_chunks])

    # Handle empty context safely
    if not context_text.strip():
        return "Error: No context available to answer the question."

    # Prompt template
    prompt = f"""
            You are a technical support assistant specialized in industrial quality processes.
            Answer the question using ONLY the context below.

            If the answer is not in the context, say:
            "I don't have enough information in the provided context."

            Context:
            {context_text}

            Question:
            {question}

            Instructions:
            - Be precise.
            - Use technical language when relevant.
            - Do not invent information.
            - Always refer to the context for your answer.
            - If the context is insufficient, clearly state that you cannot answer.
            - Answer in Spanish, unless the question is in English, then answer in English.
            - Include the document name in the answer if the information is extracted from a specific document.
            
            Answer.
            """

    return prompt