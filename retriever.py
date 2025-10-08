# retriever.py
from langchain_community.vectorstores.faiss import FAISS
from config import TOP_K
# get_embeddings remains the same, assuming it returns the embedding function
from embeddings import get_embeddings 

def get_retriever(documents):
    """
    Creates and returns a FAISS-based retriever from the provided documents.
    
    NOTE: FAISS is an in-memory index. Documents must be loaded and passed 
    to this function on every run (or cached with @st.cache_resource).
    """
    if not documents:
        raise ValueError("Documents list is empty. Cannot create FAISS index.")
    
    # Get the embedding function (only once)
    embeddings = get_embeddings()

    # Create the FAISS index from documents (re-indexes every run)
    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    # Return the retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return retriever

# IMPORTANT: You must update the part of your main application (e.g., app.py 
# or a rag_chain setup file) that calls this function to handle document 
# loading and pass the 'documents' list.

