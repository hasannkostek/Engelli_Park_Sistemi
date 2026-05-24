# =============================================================
#   Engelli Park Ihlal Tespit Sistemi
#   Model: best.pt (YOLO ozel modeli)
#   Siniflar: arac, plaka, engelli_karti
# =============================================================

import os
import time
import datetime
import cv2  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]

# ─────────────────────────────────────────────
# Ayarlar
# ─────────────────────────────────────────────
MODEL_YOLU      = "best.pt"         # Egitilmis YOLO model dosyasi
GUVEN_ESIGI     = 0.15              # conf=0.15 → dusuk esik, uzak nesneler icin
VIDEO_KAYNAK    = "test_video.mp4"  # Islenecek video dosyasi
IHLAL_KLASORU   = "ihlaller"        # Ihlal fotograflarinin kaydedilecegi klasor
COOLDOWN_SURE   = 5                 # Iki kayit arasindaki minimum bekleme suresi (saniye)

# ─────────────────────────────────────────────
# Renk tanımlamaları (BGR formatı)
# ─────────────────────────────────────────────
SARI    = (0, 255, 255)   # Röntgen yazısı
KIRMIZI = (0, 0, 255)     # İhlal uyarısı
YESIL   = (0, 220, 80)    # Uygun park bildirimi
BEYAZ   = (255, 255, 255)
SIYAH   = (0, 0, 0)

# ─────────────────────────────────────────────
# Font ayarları
# ─────────────────────────────────────────────
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_KUCUK     = 0.65
FONT_BUYUK     = 1.4
KALINLIK_INCE  = 2
KALINLIK_KALIN = 4


def metin_golge_ile_yaz(frame, metin, konum, font_olcek, renk, kalinlik):
    # Metnin okunabilirligini artirmak icin once siyah golge,
    # sonra asil renkte yazi cizer.
    x, y = konum
    # Siyah gölge (2 piksel kaydırılmış)
    cv2.putText(frame, metin, (x + 2, y + 2),
                FONT, font_olcek, SIYAH, kalinlik + 2, cv2.LINE_AA)
    # Asıl renkli metin
    cv2.putText(frame, metin, (x, y),
                FONT, font_olcek, renk, kalinlik, cv2.LINE_AA)


def ana_dongu():
    # Kamerayi acar, her kare icin YOLO tahmini yapar ve
    # ihlal durumuna gore ekrana uyari yazar.
    # ── Model yükleniyor ──────────────────────────────────
    print("[BİLGİ] Model yükleniyor:", MODEL_YOLU)
    model = YOLO(MODEL_YOLU)
    print("[BILGI] Model basariyla yuklendi.")

    # ── ihlaller klasoru olusturuluyor (yoksa) ────────────
    os.makedirs(IHLAL_KLASORU, exist_ok=True)
    print(f"[BILGI] Ihlal klasoru hazir: {IHLAL_KLASORU}/")

    # ── Cooldown: son kayit zamani ────────────────────────
    # Son fotografin cekildigi zaman. Baslangicta hic cekilmemis sayilir.
    son_kayit_zamani = 0.0

    # ── Video dosyasi aciliyor ────────────────────────────
    kamera = cv2.VideoCapture(VIDEO_KAYNAK)
    if not kamera.isOpened():
        print("[HATA] Video dosyasi acilamadi:", VIDEO_KAYNAK)
        return

    print("[BILGI] Video acildi:", VIDEO_KAYNAK, "| Cikis icin 'q' tusuna basin.")

    while True:
        basarili, kare = kamera.read()
        if not basarili:
            print("[HATA] Kare okunamadı, döngü sonlandırılıyor.")
            break

        # ── KURAL 1: YOLO Tahmini ─────────────────────────
        # conf=0.15 → uzak/bulanık nesneleri de yakalayabilmek için düşük eşik
        # imgsz parametresi KESİNLİKLE KULLANILMIYOR (görüntüyü bozmamak için)
        sonuclar = model.predict(
            source=kare,
            conf=GUVEN_ESIGI,
            verbose=False   # Konsolu gereksiz çıktıyla doldurmamak için
        )

        # ── KURAL 2: Tespit edilen sınıf isimlerini topla ─
        # .strip().lower() ile büyük/küçük harf ve boşluk hatalarını temizle
        tespit_listesi = []
        for sonuc in sonuclar:
            for kutu in sonuc.boxes:
                sinif_id  = int(kutu.cls[0])
                sinif_adi = model.names[sinif_id].strip().lower()  # Temizle
                tespit_listesi.append(sinif_adi)

                # Tespit kutusunu ekrana çiz
                x1, y1, x2, y2 = map(int, kutu.xyxy[0])
                guven      = float(kutu.conf[0])
                etiket     = f"{sinif_adi} {guven:.2f}"

                # Kutu rengi: engelli_karti → yeşil, diğerleri → turuncu
                kutu_rengi = YESIL if sinif_adi == "engelli_karti" else (255, 140, 0)
                cv2.rectangle(kare, (x1, y1), (x2, y2), kutu_rengi, 2)
                cv2.putText(kare, etiket, (x1, y1 - 8),
                            FONT, 0.55, kutu_rengi, 2, cv2.LINE_AA)

        # ── Röntgen satırı: Sol üste sarı renkle yaz ──────
        if tespit_listesi:
            rontgen_metni = "Rontgen: " + ", ".join(tespit_listesi)
        else:
            rontgen_metni = "Rontgen: hicbir sey yok"

        metin_golge_ile_yaz(kare, rontgen_metni, (15, 35),
                            FONT_KUCUK, SARI, KALINLIK_INCE)

        # ── KURAL 3: İhlal Mantığı ────────────────────────
        # Kontrol koşulu: ekranda 'arac' VEYA 'plaka' görünüyor mu?
        arac_var  = "arac"  in tespit_listesi
        plaka_var = "plaka" in tespit_listesi
        kart_var  = "engelli_karti" in tespit_listesi

        kontrol_baslasin = arac_var or plaka_var

        if kontrol_baslasin:
            if not kart_var:
                # Arac/Plaka var, engelli karti YOK → IHLAL
                metin_golge_ile_yaz(
                    kare,
                    "IHLAL: KART YOK!",
                    (40, kare.shape[0] // 2),
                    FONT_BUYUK,
                    KIRMIZI,
                    KALINLIK_KALIN
                )

                # ── EDS Kayit Sistemi ─────────────────────
                # Cooldown bitti mi kontrol et (son kayitten bu yana 5 sn gectiyse)
                simdi = time.time()
                if simdi - son_kayit_zamani >= COOLDOWN_SURE:
                    # Zaman damgali dosya adi olustur
                    damga      = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    dosya_adi  = os.path.join(IHLAL_KLASORU, f"ihlal_{damga}.jpg")
                    cv2.imwrite(dosya_adi, kare)  # Cizilmis kareyi kaydet
                    son_kayit_zamani = simdi
                    print(f"[IHLAL] Fotograf kaydedildi: {dosya_adi}")
            else:
                # Araç/Plaka var ve engelli kartı DA VAR → Uygun
                metin_golge_ile_yaz(
                    kare,
                    "PARK UYGUN (KART VAR)",
                    (40, kare.shape[0] // 2),
                    FONT_BUYUK,
                    YESIL,
                    KALINLIK_KALIN
                )

        # ── Çıkış tuşu bilgisi ────────────────────────────
        cv2.putText(kare, "Cikis: Q", (15, kare.shape[0] - 15),
                    FONT, 0.5, BEYAZ, 1, cv2.LINE_AA)

        # ── Görüntüyü ekranda göster ──────────────────────
        cv2.imshow("Engelli Park Tespit Sistemi", kare)

        # Video bitince veya 'q' tusuna basilinca cik
        # waitKey(30) → ~33 fps hizinda normal oynatma saglar
        tus = cv2.waitKey(30) & 0xFF
        if tus == ord('q'):
            print("[BILGI] Kullanici cikis yapti.")
            break

    # ── Kaynakları serbest bırak ──────────────────────────
    kamera.release()
    cv2.destroyAllWindows()
    print("[BİLGİ] Kamera kapatıldı, program sonlandı.")


# ─────────────────────────────────────────────
# Giriş noktası
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ana_dongu()