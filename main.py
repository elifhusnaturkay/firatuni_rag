import fitz  # PyMuPDF kütüphanesi
import os

PDF_FOLDER = "pdf_files"
OUTPUT_FOLDER = "ocr_outputs" # İsim aynı kalabilir, sorun değil

# Çıktı klasörünü oluştur (yoksa)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text_from_pdf():
    print("PDF'lerden metin çıkarma işlemi başlatılıyor (Yeni Yöntem: Doğrudan Okuma)...")
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            try:
                # PDF dosyasını aç
                doc = fitz.open(pdf_path)
                full_text = ""
                # Her bir sayfayı gez ve metnini al
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    full_text += page.get_text("text") + "\n\n" # Her sayfa sonuna boşluk ekle

                # Çıktı dosyasının yolunu belirle
                base_name = os.path.splitext(filename)[0]
                txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

                # Metni dosyaya yaz
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_text)

                print(f"BAŞARILI: '{filename}' dosyasının metni çıkarıldı ve '{txt_path}' dosyasına kaydedildi.")

            except Exception as e:
                print(f"HATA: '{filename}' dosyası işlenirken bir hata oluştu: {e}")

if __name__ == "__main__":
    if not os.path.isdir(PDF_FOLDER) or not os.listdir(PDF_FOLDER):
        print(f"Hata: '{PDF_FOLDER}' klasörü bulunamadı veya içi boş.")
    else:
        extract_text_from_pdf()
        print("\nTüm PDF'ler başarıyla işlendi.")