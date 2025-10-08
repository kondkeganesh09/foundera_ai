# prompt_template.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
RAG_PROMPT = """You are a helpful AI assistant. Use ONLY the provided context to answer the user's question.
If the answer cannot be found in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer concisely and include source references when possible.
"""

def get_prompt():
    return PromptTemplate(input_variables=["context", "question"], template=RAG_PROMPT)