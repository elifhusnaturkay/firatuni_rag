# ingest.py
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# --- DÜZELTME BURADA ---
# Hatalı import: from langchain_text_splitters import Document
# Doğru import:
from langchain_core.documents import Document
# -------------------------
import re

TXT_FOLDER = "ocr_outputs"
CHUNKS_FILE = "chunks.txt" # Kontrol için bu dosyayı yine de oluşturacağız
DB_DIR = "./db"

def create_chunks_from_txt(folder_path):
    print(f"'{folder_path}' klasöründeki .txt dosyaları okunuyor ve paragraflara ayrılıyor...")
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Metni çift yeni satıra göre bölerek paragrafları bul
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                # Çok kısa veya boş paragrafları atla
                if len(para.strip()) > 30: # Daha anlamlı chunklar için karakter limitini artıralım
                    # Her paragrafı bir LangChain Document nesnesine çevir
                    all_chunks.append(Document(page_content=para.strip(), metadata={"source": filename}))
    
    print(f"Toplam {len(all_chunks)} anlamlı paragraf (chunk) bulundu.")
    
    # Kontrol amacıyla chunk'ları bir dosyaya yazalım
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.page_content + "\n\n---\n\n")
    print(f"Kontrol için metin parçaları '{CHUNKS_FILE}' dosyasına yazıldı.")
    
    return all_chunks

def create_and_persist_db(chunks):
    if not chunks:
        print("Veritabanına eklenecek chunk bulunamadı.")
        return

    print("Vektör veritabanı (ChromaDB) oluşturuluyor...")
    embedding_function = SentenceTransformerEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=DB_DIR
    )
    print(f"Veritabanı başarıyla oluşturuldu ve '{DB_DIR}' klasörüne kaydedildi.")

if __name__ == '__main__':
    chunks = create_chunks_from_txt(TXT_FOLDER)
    create_and_persist_db(chunks)
    print("\nVeri hazırlama işlemi tamamlandı!")