# ♿🚗 Akıllı Engelli Otopark İhlal Tespit Sistemi 
Bu proje, engelli otopark alanlarındaki kural ihlallerini otonom olarak tespit etmek için geliştirilmiş gerçek zamanlı bir Bilgisayarlı Görü ve Elektronik Denetleme Sistemidir. 

Sistem, otopark alanını sürekli izler, araçları ve engelli park kartlarını tespit eder. Kartı olmayan bir araç park ettiğinde ihlali algılar ve zaman damgalı bir fotoğraf çekerek sisteme kaydeder.

## 🚀 Özellikler 
* **Gerçek Zamanlı Nesne Tespiti:** YOLOv11 mimarisi kullanılarak yüksek hızlı ve hassas tespit.
* **Özel Veri Seti:** Roboflow üzerinden etiketlenip hazırlanan özel model (`best.pt`).
* **Akıllı EDS Mantığı:** Araç ve plaka tespiti yapıldığında "Engelli Kartı" kontrolü sağlayan algoritmik yapı.
* **Otonom Kayıt ve Cooldown:** İhlal durumunda 5 saniyelik bekleme süresi ile spam önleyici otomatik fotoğraf loglama sistemi.

## 🛠️ Kullanılan Teknolojiler
* **Yapay Zeka & Derin Öğrenme:** Python, Ultralytics YOLOv11
* **Görüntü İşleme:** OpenCV
* **Veri Seti & Eğitim:** Roboflow, Google Colab (T4 GPU)

## ⚙️ Kurulum ve Kullanım

1. Depoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/Engelli_Park_Sistemi.git](https://github.com/KULLANICI_ADIN/Engelli_Park_Sistemi.git)

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt

3. Sistemi çalıştırın (Varsayılan olarak test_video.mp4 üzerinden çalışır):
   ```bash
   python main.py


**Not: Kameradan canlı test yapmak için main.py içindeki cv2.VideoCapture("test_video.mp4") satırını cv2.VideoCapture(0) olarak değiştirebilirsiniz.**

Yazar: **Hasan Köstek**
