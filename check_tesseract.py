import pytesseract
import os

# Windows kullanıcıları için Tesseract'ın kurulu olduğu yeri belirtmek gerekebilir.
# Eğer kurulum sırasında varsayılan yola kurduysanız, bu satırı kullanabilirsiniz.
# Başka bir yere kurduysanız, yolu ona göre güncelleyin.
if os.name == 'nt': # Eğer işletim sistemi Windows ise
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract başarıyla bulundu! Versiyon: {version}")
except pytesseract.TesseractNotFoundError:
    print("HATA: Tesseract bulunamadı!")
    print("Lütfen Tesseract-OCR'ı bilgisayarınıza kurduğunuzdan ve gerekirse yukarıdaki script'te yolunu doğru belirttiğinizden emin olun.")