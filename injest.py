import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

DOCUMENTS_DIR = "cleaned-test-transcripts"

## --- Chunking ---

def load_and_chunk(directory, chunk_size=500, overlap=50):
    """Read all text files and split them into overlapping chunks."""
    chunks = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            text = f.read()

        # Simple character-based chunking with overlap
        for start in range(0, len(text), chunk_size - overlap):
            chunk_text = text[start : start + chunk_size]
            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": filename,
                        "start": start,
                    }
                )
    return chunks


## --- Embedding and Storage ---

def embed_and_store(chunks):
    """Embed chunks with OpenAI and store in ChromaDB."""
    openai = OpenAI()
    db = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if re-running
    try:
        db.delete_collection("documents")
    except Exception:
        pass

    collection = db.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )

    # OpenAI accepts batches of up to 2048 texts
    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]

        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = [item.embedding for item in response.data]

        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": c["source"], "start": c["start"]} for c in batch],
        )

    print(f"Stored {len(chunks)} chunks from {len(set(c['source'] for c in chunks))} documents")
    return collection


if __name__ == "__main__":
    chunks = load_and_chunk(DOCUMENTS_DIR)
    embed_and_store(chunks)
    