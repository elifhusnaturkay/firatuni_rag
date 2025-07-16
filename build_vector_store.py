import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

embedding_model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

TEXT_FOLDER = "ocr_outputs"
CHUNKS_FILE = "chunks.txt"
INDEX_FILE = "vector_index.index"

def create_chunks():
    all_chunks = []
    print(f"'{TEXT_FOLDER}' klasöründeki metinler okunuyor...")
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
                full_text = f.read()
                
                paragraphs = re.split(r'\n\s*\n', full_text)
                
                for para in paragraphs:
                    if len(para.strip()) > 20:
                        cleaned_para = re.sub(r'---\s*Sayfa\s*\d+\s*---', '', para).strip()
                        if cleaned_para:
                            all_chunks.append(cleaned_para)
    
    print(f"Toplam {len(all_chunks)} anlamlı metin parçası (chunk) bulundu.")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_chunks))
    print(f"Metin parçaları '{CHUNKS_FILE}' dosyasına yazıldı.")
    return all_chunks

def create_vector_store(chunks):
    print(f"Embedding modeli yükleniyor: '{embedding_model_name}'...")
    model = SentenceTransformer(embedding_model_name)
    
    print("Metin parçaları vektörlere dönüştürülüyor (embedding)...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    print("FAISS vektör veritabanı oluşturuluyor...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    print(f"Vektör veritabanı başarıyla '{INDEX_FILE}' dosyasına kaydedildi.")

if __name__ == "__main__":
    chunks = create_chunks()
    if chunks:
        create_vector_store(chunks)
    else:
        print("İşlenecek metin parçası bulunamadı.")