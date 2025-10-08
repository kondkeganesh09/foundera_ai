import traceback
from rag_chain import ManualRAG

def main():
    # Initialize RAG
    qa = ManualRAG()
    print("RAG chain loaded with Hugging Face LLM. Ready to answer your questions!\n")

    while True:
        query = input("Enter your question (or 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            result = qa({"query": query})
            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            print("\n ANSWER:\n", answer)
            print("\n SOURCES:")
            if sources:
                for idx, doc in enumerate(sources, 1):
                    source_id = doc.metadata.get("source") or doc.metadata.get("id") or f"doc{idx}"
                    snippet = doc.page_content[:200].replace("\n", " ") + "..."
                    print(f"{idx}. {source_id} | {snippet}")
            else:
                print("No sources found.")
            print("\n" + "-"*60 + "\n")

        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
