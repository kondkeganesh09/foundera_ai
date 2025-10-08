
# rag_chain.py
import os
import requests
from retriever import get_retriever
from config import TOP_K, PERSIST_DIR, COLLECTION_NAME
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from embeddings import get_embeddings
from langchain.prompts import PromptTemplate


class ManualRAG:
    def __init__(self):
        self.retriever = get_retriever()

        # Chroma vector DB
        self.vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
            embedding_function=get_embeddings()
        )

        # Hugging Face Router API setup
        self.HF_TOKEN = "XXXXXXX"  
        self.API_URL = "https://router.huggingface.co/v1/chat/completions"
        self.MODEL = "Qwen/Qwen2.5-72B-Instruct:together"  # or another Qwen provider
        self.HEADERS = {"Authorization": f"Bearer {self.HF_TOKEN}"}

        # Prompt template
        self.prompt = PromptTemplate(
            template=(
                "You are an assistant. Use the following documents:\n\n{context}\n\n"
                "Question: {question}\nAnswer:"
            ),
            input_variables=["context", "question"],
        )

        print("ManualRAG initialized successfully with Qwen model.")

    def _call_hf_model(self, prompt: str) -> str:
        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 512,
            "temperature": 0.0
        }
        resp = requests.post(self.API_URL, headers=self.HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(data)

    def run(self, query: str):
        docs = self.retriever.invoke(query)

        if len(docs) == 0:
            all_docs = self.vectordb.get(include=["documents", "metadatas"])
            matched_docs = []
            query_norm = query.lower()
            for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                doc_text = doc.lower()
                meta_text = " ".join([str(v).lower() for v in meta.values()])
                if query_norm in doc_text or query_norm in meta_text:
                    matched_docs.append(Document(page_content=doc, metadata=meta))
            docs = matched_docs

        print(f"[DEBUG] Retrieved {len(docs)} docs for query: '{query}'")

        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        answer = (
            "No relevant documents found."
            if not context
            else self._call_hf_model(self.prompt.format(context=context, question=query))
        )

        return {"result": answer, "source_documents": docs}

    def __call__(self, inputs: dict):
        query = inputs.get("query", "")
        return self.run(query)
