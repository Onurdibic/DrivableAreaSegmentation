import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
import os
from transformers import SegformerConfig

# Cihazı belirleme
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {cihaz}")

# Modeli yükleme
def egitilmiş_modeli_yükle(model_yolu):
    config = SegformerConfig.from_pretrained("C:/Users/onurd/Desktop/archive/segformer_offline/config.json")
    config.num_labels = 3  # 3 sınıf: background, drivable, lane
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(torch.load(model_yolu, map_location=cihaz))
    model.to(cihaz)
    model.eval()
    return model

# Görseli dönüştürme
def görseli_isle(görsel_yolu, boyut=(640, 640)):
    görsel = Image.open(görsel_yolu).convert("RGB")
    görsel_büyütülmüş = görsel.resize(boyut, Image.BILINEAR)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(görsel_büyütülmüş).unsqueeze(0), görsel_büyütülmüş

# Ground truth maskelerini yükle ve birleştir
def maskeleri_yükle_ve_birleştir(görsel_dosyası_adi, boyut=(640, 640)):
    görsel_id = os.path.splitext(görsel_dosyası_adi)[0]
    temel_yol = "C:/Users/onurd/Desktop/archive/val/masks"
    drivable_yol = os.path.join(temel_yol, f"{görsel_id}_drivable_mask.png")
    lane_yol = os.path.join(temel_yol, f"{görsel_id}_lane_mask.png")

    if not os.path.exists(drivable_yol) or not os.path.exists(lane_yol):
        print(f"Maske dosyaları bulunamadı: {görsel_dosyası_adi}")
        return None

    drivable_mask = Image.open(drivable_yol).convert("L").resize(boyut, Image.NEAREST)
    lane_mask = Image.open(lane_yol).convert("L").resize(boyut, Image.NEAREST)

    drivable_np = np.array(drivable_mask)
    lane_np = np.array(lane_mask)

    birleşik_mask = np.zeros_like(drivable_np)
    birleşik_mask[drivable_np > 0] = 1
    birleşik_mask[lane_np > 0] = 2  # lane üst üste yazılacak

    return birleşik_mask

# Metrik hesaplama
def metrikleri_hesapla(pred_mask_büyütülmüş, gt_mask_büyütülmüş):
    TP = np.sum((pred_mask_büyütülmüş == 1) & (gt_mask_büyütülmüş == 1))
    FP = np.sum((pred_mask_büyütülmüş == 1) & (gt_mask_büyütülmüş != 1))
    TN = np.sum((pred_mask_büyütülmüş == 0) & (gt_mask_büyütülmüş == 0))
    FN = np.sum((pred_mask_büyütülmüş == 0) & (gt_mask_büyütülmüş == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, specificity, iou

# Bir görseli işleme ve kaydetme
def isle_ve_kaydet(görsel_yolu, model, çıktı_klasörü, alpha=0.5):
    görsel_tensor, büyütülmüş_görsel = görseli_isle(görsel_yolu)
    görsel_tensor = görsel_tensor.to(cihaz)

    with torch.no_grad():
        çıktı = model(pixel_values=görsel_tensor).logits

    pred_mask = torch.argmax(çıktı, dim=1).squeeze(0).cpu().numpy()
    pred_mask_büyütülmüş = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(büyütülmüş_görsel.size, Image.NEAREST))

    görsel_dosyası_adi = os.path.basename(görsel_yolu)
    gt_mask = maskeleri_yükle_ve_birleştir(görsel_dosyası_adi)

    if gt_mask is None:
        return

    gt_mask_büyütülmüş = np.array(Image.fromarray(gt_mask.astype(np.uint8)).resize(büyütülmüş_görsel.size, Image.NEAREST))  # Burada eksik olan dönüşümü ekliyoruz.

    doğru_pikseller = np.sum(pred_mask_büyütülmüş == gt_mask_büyütülmüş)
    toplam_pikseller = pred_mask_büyütülmüş.size
    doğruluk = (doğru_pikseller / toplam_pikseller) * 100

    precision, recall, specificity, iou = metrikleri_hesapla(pred_mask_büyütülmüş, gt_mask_büyütülmüş)

    # Sınıf 1 (drivable area) doğruluğu
    drivable_mask = gt_mask_büyütülmüş == 1
    drivable_toplam = drivable_mask.sum()
    drivable_doğru = np.sum((pred_mask_büyütülmüş == 1) & drivable_mask)
    drivable_doğruluk = (drivable_doğru / drivable_toplam) * 100 if drivable_toplam > 0 else 0

    # Sınıf 2 (lane) doğruluğu
    lane_mask = gt_mask_büyütülmüş == 2
    lane_toplam = lane_mask.sum()
    lane_doğru = np.sum((pred_mask_büyütülmüş == 2) & lane_mask)
    lane_doğruluk = (lane_doğru / lane_toplam) * 100 if lane_toplam > 0 else 0

    genişlik, yükseklik = büyütülmüş_görsel.size

    renk_haritası = {
        1: [0, 255, 0],     # Yol - Yeşil
        2: [255, 0, 0]      # Şerit - Kırmızı
    }

    seg_img = np.zeros((yükseklik, genişlik, 4), dtype=np.uint8)
    for sınıf_id, renk in renk_haritası.items():
        seg_img[pred_mask_büyütülmüş == sınıf_id, :3] = renk
        seg_img[pred_mask_büyütülmüş == sınıf_id, 3] = 255

    orijinal_img = np.array(büyütülmüş_görsel)
    harmanlanmış = orijinal_img.copy()
    for c in range(3):
        harmanlanmış[:, :, c] = (alpha * seg_img[:, :, c] + (1 - alpha) * orijinal_img[:, :, c]).astype(np.uint8)

    # PIL imaja çevirip üzerine metin yazacağız
    harmanlanmış_pil = Image.fromarray(harmanlanmış)
    draw = ImageDraw.Draw(harmanlanmış_pil)

    # Yazı fontu
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    metin = (
        f"Doğruluk: {doğruluk:.2f}%\n" #doğru tahminler modelin doğru sınıflandırdığı pikselleri, toplam tahminler ise modelin tüm tahminlerini ifade eder.
        f"Keskinlik: {precision * 100:.2f}%\n"#modelin pozitif olarak tahmin ettiği her şeyin gerçekten doğru olma olasılığını gösterir.
        f"Geri Çağırma: {recall * 100:.2f}%\n"
        f"Özgüllük: {specificity * 100:.2f}%\n"
        f"IoU: {iou * 100:.2f}%\n" #Intersection over Union ,birlik üzerinde kesişme
        f"Drivable Acc: {drivable_doğruluk:.2f}%\n"
        f"Lane Acc: {lane_doğruluk:.2f}%"
    )

    draw.multiline_text((10, 10), metin, fill=(255, 255, 255), font=font)

    # Kaydet
    os.makedirs(çıktı_klasörü, exist_ok=True)
    çıktı_yolu = os.path.join(çıktı_klasörü, görsel_dosyası_adi)
    harmanlanmış_pil.save(çıktı_yolu)
    print(f"Kaydedildi: {çıktı_yolu}")

# --- Buradan sonrası çalıştırılıyor ---

# Modeli yükle
model_yolu = "C:/Users/onurd/Desktop/archive/best_segformer_model01.pth"
model = egitilmiş_modeli_yükle(model_yolu)

# Görsellerin bulunduğu klasör
girdi_klasörü = "C:/Users/onurd/Desktop/archive/val/images"
çıktı_klasörü = "C:/Users/onurd/Desktop/archive/segmentation_results"

# Tüm görselleri işle
for dosya_adi in os.listdir(girdi_klasörü):
    if dosya_adi.lower().endswith((".jpg", ".png", ".jpeg")):
        görsel_yolu = os.path.join(girdi_klasörü, dosya_adi)
        isle_ve_kaydet(görsel_yolu, model, çıktı_klasörü)
