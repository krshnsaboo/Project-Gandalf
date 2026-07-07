from rag_pipeline import RAGPipeline


def main():

    print("=" * 70)
    print("        Striver A2Z DSA Lecture Navigator")
    print("=" * 70)

    pipeline = RAGPipeline(
        embeddings_path="lecture_embeddings/all_lecture_embeddings.pkl",
        index_path="lecture_embeddings/faiss_index.bin"
    )

    while True:

        query = input("\nAsk a question (type 'exit' to quit): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nGoodbye!")
            break

        if not query:
            continue

        print("\nSearching...\n")

        try:
            response = pipeline.search(query)

            print("=" * 70)
            print(response)
            print("=" * 70)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()