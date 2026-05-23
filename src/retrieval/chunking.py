from langchain_text_splitters import RecursiveCharacterTextSplitter

""" Chunking process with overlap """
def chunk_text(text, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    return splitter.split_text(text)
