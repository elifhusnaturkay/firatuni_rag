import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
INPUT_DIR = "ocr_outputs"

# Yerel embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")  # hızlı ve hafif

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_all_texts():
    all_embeddings = []
    all_chunks = []

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()

            chunks = chunk_text(text)
            for chunk in chunks:
                try:
                    emb = model.encode(chunk)
                    all_embeddings.append(emb)
                    all_chunks.append(chunk)
                except Exception as e:
                    print(f"Hata: {e}")

    return all_chunks, np.array(all_embeddings).astype("float32")

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, "vector_index.index")
    print(" FAISS index kaydedildi: vector_index.index")

if __name__ == "__main__":
    chunks, embeddings = process_all_texts()
    build_faiss_index(embeddings)

    # chunks.txt dosyasına kaydet
    with open("chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(chunks))
