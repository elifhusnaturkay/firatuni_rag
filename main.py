from pdf2image import convert_from_path
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

poppler_path = r'C:\poppler\poppler-24.08.0\Library\bin'

os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'


PDF_FOLDER = "pdf_files"
OUTPUT_FOLDER = "ocr_outputs"
LANG = "tur"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    print("PDF sayfaları görsele dönüştürülüyor...")
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    full_text = ""
    print(f"{len(pages)} sayfa bulundu. OCR işlemi başlıyor...")
    for i, page in enumerate(pages):

        text = pytesseract.image_to_string(page, lang=LANG)
        full_text += f"\n\n--- Sayfa {i+1} ---\n{text}"
        print(f"Sayfa {i+1} metne çevrildi.")
    return full_text

def run_ocr_on_folder():
    pdf_found = False
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_found = True
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"------------------------------------------")
            print(f"OCR başlatılıyor: {filename}")
            text = extract_text_from_pdf(pdf_path)

            base_name = os.path.splitext(filename)[0]
            if base_name.lower().endswith(".pdf"):
                 base_name = os.path.splitext(base_name)[0]

            txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"BAŞARILI: OCR tamamlandı ve metin şu dosyaya kaydedildi: {txt_path}")
            print(f"------------------------------------------\n")

    if not pdf_found:
        print(f"UYARI: '{PDF_FOLDER}' klasöründe işlenecek .pdf uzantılı bir dosya bulunamadı.")


if __name__ == "__main__":
    if not os.path.isdir(PDF_FOLDER):
        print(f"HATA: '{PDF_FOLDER}' adında bir klasör bulunamadı. Lütfen oluşturun ve içine PDF dosyalarınızı koyun.")
    else:
        run_ocr_on_folder()