import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import time  

# Cihazı belirleme
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {cihaz}")

def modeli_yukle(model_dosyasi):
    ayar = SegformerConfig.from_pretrained("C:/Users/onurd/Desktop/archive/segformer_offline/config.json")
    ayar.num_labels = 3
    model = SegformerForSemanticSegmentation(ayar)
    model.load_state_dict(torch.load(model_dosyasi, map_location=cihaz))
    model.to(cihaz)
    model.eval()
    return model

# Frame ön işleme
def frame_onisleme(frame, boyut=(640, 640)):
    goruntu = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    goruntu = goruntu.resize(boyut, Image.BILINEAR)
    donusum = transforms.Compose([transforms.ToTensor()])
    return donusum(goruntu).unsqueeze(0).to(cihaz), np.array(goruntu)

# Segmentasyon tahmini
def segmentasyon_tahmin(model, frame):
    giris, yeniden_boyutlandirilmis = frame_onisleme(frame)
    with torch.no_grad():
        cikti = model(pixel_values=giris).logits
    tahmin_maske = torch.argmax(cikti, dim=1).squeeze(0).cpu().numpy()
    return tahmin_maske, yeniden_boyutlandirilmis

# Renk haritası
renk_haritasi = {
    1: [0, 255, 0],   # Yol - Yeşil
    2: [255, 0, 0]    # Şerit - Kırmızı
}

# Maskeyi renklendir
def renkli_maske_olustur(maske):
    yukseklik, genislik = maske.shape
    seg_goruntu = np.zeros((yukseklik, genislik, 3), dtype=np.uint8)
    for sinif_id, renk in renk_haritasi.items():
        seg_goruntu[maske == sinif_id] = renk
    return seg_goruntu

# Ana video işleme fonksiyonu
def video_isle(video_yolu, model):
    video = cv2.VideoCapture(video_yolu)

    if not video.isOpened():
        print("Video açılamadı.")
        return

    while True:
        basarili_mi, frame = video.read()
        if not basarili_mi:
            break

        # İşleme başlangıç zamanı
        islem_baslangic = time.time()

        tahmin_maske, yeniden_boyutlandirilmis = segmentasyon_tahmin(model, frame)
        renkli_maske = renkli_maske_olustur(tahmin_maske)

        if renkli_maske.shape[:2] != yeniden_boyutlandirilmis.shape[:2]:
            renkli_maske = cv2.resize(renkli_maske, (yeniden_boyutlandirilmis.shape[1], yeniden_boyutlandirilmis.shape[0]))

        saydamlik = 0.5
        birlesik = cv2.addWeighted(yeniden_boyutlandirilmis, 1 - saydamlik, renkli_maske, saydamlik, 0)

        birlesik_bgr = cv2.cvtColor(birlesik, cv2.COLOR_RGB2BGR)

        # İşleme bitiş zamanı
        islem_bitis = time.time()
        gecen_sure = islem_bitis - islem_baslangic

        # FPS hesapla
        if gecen_sure > 0:
            fps = 1 / gecen_sure
            cv2.putText(birlesik_bgr, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Segmentasyon", birlesik_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Modeli yükle
model_yolu = "C:/Users/onurd/Desktop/archive/best_segformer_model01.pth"
model = modeli_yukle(model_yolu)

# Video yolu
video_yolu = "C:/Users/onurd/Desktop/sehir.mp4"

# Videoyu işle
video_isle(video_yolu, model)
