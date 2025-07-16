# ask.py
import torch
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import CrossEncoder
import sys
import codecs

# --- Windows için Türkçe Karakter Çözümü ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# --- Modelleri Yükle ---
print("Modeller yükleniyor...")
embedding_function = SentenceTransformerEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# --- DEĞİŞİKLİK BURADA: YENİ VE DAHA AKILLI BİR ÜRETKEN MODEL ---
device = "cuda" if torch.cuda.is_available() else "cpu"
llm_model_name = "google/gemma-2b-it" # DeepSeek yerine Google'ın Gemma modelini kullanıyoruz
# -------------------------------------------------------------

print(f"Yeni üretken model yükleniyor: {llm_model_name}. Bu işlem biraz zaman alabilir...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
# Gemma için torch_dtype=torch.float16 eklemek performansı artırabilir
model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16).to(device)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=device)

print("Reranker modeli yükleniyor...")
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Tüm modeller başarıyla yüklendi.")

# --- Veritabanını Yükle ---
print("Vektör veritabanı yükleniyor...")
db = Chroma(persist_directory="./db", embedding_function=embedding_function)
print("Veritabanı yüklendi.")

# --- Prompt Şablonu ---
prompt_template = """
### GÖREV TANIMI ###
Sen, Fırat Üniversitesi yönetmelikleri konusunda uzman bir yardımcı asistansın. Görevin, sana verilen "İLGİLİ BİLGİLER" bölümündeki metinlere dayanarak kullanıcının "SORU"sunu yanıtlamaktır.

### KURALLAR ###
1. Cevabını SADECE ve SADECE sana verilen "İLGİLİ BİLGİLER" içindeki bilgileri kullanarak oluştur.
2. Bilgiler yeterli değilse veya sorunun cevabı metinlerde yoksa, "Bu soruya yanıt verecek bilgi, bana sağlanan belgelerde bulunmamaktadır." de.
3. Asla tahmin yapma veya metin dışı bilgi kullanma.
4. Cevabını net, anlaşılır ve doğrudan sorulan soruya odaklı olmalı.

### İLGİLİ BİLGİLER ###
{context}

### SORU ###
{query}

### CEVAP ###
"""

print("\nTüm sistem hazır. Şimdi soru sorabilirsiniz. Çıkmak için 'exit' yazın.")
while True:
    query = input("\nSoru: ")
    if query.lower() == 'exit':
        break

    # 1. Adım: Arama
    retrieved_docs = db.similarity_search(query, k=15)

    # 2. Adım: Reranking
    print("\nİlk arama sonuçları bulundu, şimdi Reranker ile eleniyor...")
    reranker_input_pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = reranker_model.predict(reranker_input_pairs)
    
    docs_with_scores = sorted(list(zip(scores, retrieved_docs)), key=lambda x: x[0], reverse=True)
    
    top_k_reranked = 3
    final_docs = [doc for score, doc in docs_with_scores[:top_k_reranked]]
    context = "\n\n".join([doc.page_content for doc in final_docs])

    # DEBUG
    print("\n" + "="*50)
    print(f"DEBUG: Reranker sonrası seçilen EN İYİ {top_k_reranked} metin:")
    print(context)
    print("="*50 + "\n")

    # 3. Adım: Cevap Üretme
    final_prompt = prompt_template.format(context=context, query=query)
    
    result = pipe(final_prompt)
    answer = result[0]['generated_text'].split("### CEVAP ###")[-1].strip()

    print(f" Cevap: {answer}")