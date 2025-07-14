import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Aynı embedding modeli
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk'ları da dosyadan yükleyelim
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n---\n\n")

# FAISS index'i yükle
index = faiss.read_index("vector_index.index")

# Kullanıcıdan soru al
query = input("Soru: ")
query_embedding = model.encode(query).astype("float32")

# En yakın 10 chunk'ı bul
top_k = 10
D, I = index.search(np.array([query_embedding]), top_k)

# İlgili içerikleri getir
relevant_chunks = [chunks[i] for i in I[0]]

print("\n En alakalı içerikler:\n")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"\n--- [{i}] ---\n{chunk}")

# (İsteğe bağlı: Buradan sonra OpenAI ya da başka LLM ile yanıt üretilebilir)
