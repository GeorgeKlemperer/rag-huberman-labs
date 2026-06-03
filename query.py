import sys
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import anthropic

load_dotenv()


def retrieve(question, n_results=5):
    """Find the most relevant chunks for a question."""
    openai = OpenAI()
    db = chromadb.PersistentClient(path="./chroma_db")
    collection = db.get_collection("documents")

    # Embed the question with the same model used for documents
    response = openai.embeddings.create(input=[question], model="text-embedding-3-small")
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results["documents"][0], results["metadatas"][0]


def generate(question, context_chunks, metadata):
    """Ask Claude to answer the question using the retrieved context."""
    client = anthropic.Anthropic()

    # Build the context string
    context = ""
    for i, (chunk, meta) in enumerate(zip(context_chunks, metadata)):
        context += f"\n--- Chunk {i + 1} (from {meta['source']}) ---\n{chunk}\n"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a helpful assistant. Answer questions based only on the provided context. If the context doesn't contain enough information to answer, say so.",
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )

    return response.content[0].text


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("Usage: python query.py <question>")

    print(f"Question: {question}\n")

    chunks, metadata = retrieve(question)

    print("Retrieved chunks:")
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        print(f"  [{i + 1}] {meta['source']}: {chunk[:80]}...")
    print()

    answer = generate(question, chunks, metadata)
    print(f"Answer: {answer}")
