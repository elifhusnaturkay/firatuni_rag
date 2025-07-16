import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

embedding_model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
embedding_model = SentenceTransformer(embedding_model_name)

with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n---\n\n")
index = faiss.read_index("vector_index.index")

prompt_template = """
### GÖREV TANIMI ###
Sen, Fırat Üniversitesi yönetmelikleri konusunda uzman bir yardımcı asistansın. Görevin, sana verilen "KAYNAK METİNLER" bölümündeki bilgilere dayanarak kullanıcının "SORU"sunu yanıtlamaktır.

### KURALLAR ###
1. Cevabını SADECE ve SADECE sana verilen "KAYNAK METİNLER" içindeki bilgileri kullanarak oluştur.
2. KAYNAK METİNLER dışında asla kendi bilgini kullanma veya tahmin yapma.
3. Eğer sorunun cevabı verilen metinlerde yoksa, kesinlikle "Bu soruya cevap verecek bilgi, bana sağlanan belgelerde bulunmamaktadır." yanıtını ver.
4. Cevapların net, anlaşılır ve doğrudan sorulan soruya odaklı olmalı.

### KAYNAK METİNLER ###
{context}

### SORU ###
{query}

### CEVAP ###
"""

query = input("Soru: ")
query_embedding = embedding_model.encode(query).astype("float32")

# En yakın ve en alakalı 5 metin parçasını bul
D, I = index.search(np.array([query_embedding]), k=5)
relevant_chunks = [chunks[i] for i in I[0]]
context = "\n\n".join(relevant_chunks)

final_prompt = prompt_template.format(context=context, query=query)

inputs = tokenizer(final_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0) # temperature=0.0 daha kesin cevaplar için
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

answer_only = answer.split("### CEVAP ###")[-1].strip()

print("\n Cevap:\n")
print(answer_only)