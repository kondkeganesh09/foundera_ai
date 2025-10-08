# retriever.py
from langchain_chroma import Chroma
from config import PERSIST_DIR, COLLECTION_NAME, TOP_K
from embeddings import get_embeddings

def get_retriever():
    embeddings = get_embeddings()

    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    return retriever

