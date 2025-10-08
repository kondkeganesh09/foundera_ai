# embeddings.py
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Returns the embedding function used for Chroma vector DB.
    Must match embeddings used during ingestion.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example function to embed new documents (if needed)
def embed_and_store(documents, vectordb):
    """
    documents: list of dicts {"page_content": str, "metadata": dict}
    vectordb: Chroma vectorstore instance
    """
    vectordb.add_documents(documents)
    vectordb.persist()
    print(f"Added {len(documents)} documents and persisted to Chroma.")
