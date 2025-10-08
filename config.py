# config.py
import os

# ----------------------------
# Chroma DB config
# ----------------------------
PERSIST_DIR = "./chroma_db"

COLLECTION_NAME = "schemes_collection"



# ----------------------------
# Retrieval config
# ----------------------------
TOP_K = 6  # number of documents to retrieve
SEARCH_TYPE = "similarity"  # or "mmr"