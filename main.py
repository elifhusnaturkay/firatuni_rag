from pdf2image import convert_from_path
import pytesseract
import os

PDF_FOLDER = "pdf_files"
OUTPUT_FOLDER = "ocr_outputs"
LANG = "tur"

# Çıktı klasörünü oluştur (yoksa)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path)
    full_text = ""
    for i, page in enumerate(pages):
        text = pytesseract.image_to_string(page, lang=LANG)
        full_text += f"\n\n--- Sayfa {i+1} ---\n{text}"
    return full_text

def run_ocr_on_folder():
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"OCR başlatılıyor: {filename}")
            text = extract_text_from_pdf(pdf_path)

            # txt dosya ismini belirle
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.txt")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"OCR tamamlandı: {txt_path}")

if __name__ == "__main__":
    run_ocr_on_folder()
