import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Arama motoru olarak kullandığımız GÜÇLÜ TÜRKÇE modelin burada da belirtildiğinden emin olalım.
embedding_model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
print(f"Embedding modeli (arama motoru) yükleniyor: {embedding_model_name}...")
model = SentenceTransformer(embedding_model_name)
print("Embedding modeli başarıyla yüklendi!")


with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n---\n\n")

index = faiss.read_index("vector_index.index")

query = input("Soru: ")
query_embedding = model.encode(query).astype("float32")

# En yakın 10 chunk'ı bul (daha fazla sonuç görmek için sayıyı artırdık)
top_k = 10
D, I = index.search(np.array([query_embedding]), top_k)

relevant_chunks = [chunks[i] for i in I[0]]

print("\n" + "="*50)
print("DEBUG: Soruya en alakalı bulunan metin parçaları şunlar:")
print("="*50 + "\n")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"\n--- Bulunan Parça [{i}] ---\n{chunk}")
print("\n" + "="*50)