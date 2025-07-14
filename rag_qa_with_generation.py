import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model yükle (GPU varsa kullanılır)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Aynı embedding modeli
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chunk'ları yükle
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n---\n\n")

# FAISS index’i yükle
index = faiss.read_index("vector_index.index")

# Kullanıcıdan soru al
query = input("Soru: ")
query_embedding = embedding_model.encode(query).astype("float32")

# En yakın 5 chunk’ı bul
D, I = index.search(np.array([query_embedding]), k=5)
relevant_chunks = [chunks[i] for i in I[0]]
context = "\n".join(relevant_chunks)

# Prompta bağlamla birlikte soru ver
prompt = f"""
Aşağıdaki bilgiye göre soruyu yanıtla:

{context}

Soru: {query}
Cevap:
"""

# Tokenize ve modele ver
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Cevabı yazdır
print("\n🤖 Cevap:\n")
print(answer.replace(prompt.strip(), "").strip())
