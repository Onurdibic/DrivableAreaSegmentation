import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
import os
from transformers import SegformerConfig
 	
# Cihazı belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# Modeli yükleme
def load_trained_model(model_path):
    config = SegformerConfig.from_pretrained("C:/Users/onurd/Desktop/archive/segformer_offline/config.json")
    config.num_labels = 3  # 3 sınıf: background, drivable, lane
    model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Görseli dönüştürme (640x640 boyutuna getirerek)
def preprocess_image(image_path, img_size=(640, 640)):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(img_size, Image.BILINEAR)

    transform = transforms.Compose([transforms.ToTensor()])
    
    return transform(image_resized).unsqueeze(0), image_resized

# Ground truth maskeleri yükle ve birleştir
def load_combined_ground_truth_mask(image_filename, img_size=(640, 640)):
    image_id = os.path.splitext(image_filename)[0]
    base_path = "C:/Users/onurd/Desktop/archive/val/masks"

    drivable_path = os.path.join(base_path, f"{image_id}_drivable_mask.png")
    lane_path = os.path.join(base_path, f"{image_id}_lane_mask.png")

    if not os.path.exists(drivable_path) or not os.path.exists(lane_path):
        print("Maske dosyaları bulunamadı.")
        return None

    drivable_mask = Image.open(drivable_path).convert("L").resize(img_size, Image.NEAREST)
    lane_mask = Image.open(lane_path).convert("L").resize(img_size, Image.NEAREST)

    drivable_np = np.array(drivable_mask)
    lane_np = np.array(lane_mask)

    # 0: background, 1: drivable, 2: lane
    combined_mask = np.zeros_like(drivable_np)
    combined_mask[drivable_np > 0] = 1
    combined_mask[lane_np > 0] = 2  # lane overwrite yapar

    return combined_mask

def calculate_metrics(pred_mask_resized, gt_mask_resized):
    # TP, FP, TN, FN hesaplama
    TP = np.sum((pred_mask_resized == 1) & (gt_mask_resized == 1))  # True Positive
    FP = np.sum((pred_mask_resized == 1) & (gt_mask_resized != 1))  # False Positive
    TN = np.sum((pred_mask_resized == 0) & (gt_mask_resized == 0))  # True Negative
    FN = np.sum((pred_mask_resized == 0) & (gt_mask_resized == 1))  # False Negative

    # Hesaplamalar
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, specificity, iou

def visualize_segmentation_and_accuracy(image_path, model, alpha=0.5):
    image_tensor, resized_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(pixel_values=image_tensor).logits

    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Görseli 640x640 boyutunda yeniden boyutlandır
    pred_mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(resized_image.size, Image.NEAREST))

    # Ground truth maskeyi yükle
    image_filename = os.path.basename(image_path)
    gt_mask = load_combined_ground_truth_mask(image_filename)

    if gt_mask is None:
        return

    # Genel doğruluk
    correct_pixels = np.sum(pred_mask_resized == gt_mask)
    total_pixels = pred_mask_resized.size
    accuracy = (correct_pixels / total_pixels) * 100
    print(f"Genel Doğruluk (Tüm sınıflar): {accuracy:.2f}%")

    # Precision, Recall, Specificity ve IoU hesaplama
    precision, recall, specificity, iou = calculate_metrics(pred_mask_resized, gt_mask)
    print(f"Hassasiyet (Precision): {precision * 100:.2f}%")
    print(f"Duyarlılık (Recall): {recall * 100:.2f}%")
    print(f"Keskinlik (Specificity): {specificity * 100:.2f}%")
    print(f"Intersection over Union (IoU): {iou * 100:.2f}%")

    # Sınıf 1 (drivable area) doğruluğu
    drivable_mask = gt_mask == 1
    drivable_total = drivable_mask.sum()
    drivable_correct = np.sum((pred_mask_resized == 1) & drivable_mask)
    drivable_accuracy = (drivable_correct / drivable_total) * 100 if drivable_total > 0 else 0
    print(f"Drivable Area (Sınıf 1) Doğruluk: {drivable_accuracy:.2f}%")

    # Sınıf 2 (lane) doğruluğu
    lane_mask = gt_mask == 2
    lane_total = lane_mask.sum()
    lane_correct = np.sum((pred_mask_resized == 2) & lane_mask)
    lane_accuracy = (lane_correct / lane_total) * 100 if lane_total > 0 else 0
    print(f"Lane (Sınıf 2) Doğruluk: {lane_accuracy:.2f}%")

    # Görselleştirme
    width, height = resized_image.size
    mask_resized = pred_mask_resized  # Yeniden boyutlandırılmış maskeyi kullan

    # Gerçek (ground truth) maskesinin görselleştirilmesi
    gt_mask_resized = np.array(Image.fromarray(gt_mask.astype(np.uint8)).resize(resized_image.size, Image.NEAREST))

    color_map = {
        1: [0, 255, 0],     # Yol - Yeşil
        2: [255, 0, 0]      # Şerit - Kırmızı
    }

    seg_img = np.zeros((height, width, 4), dtype=np.uint8)
    for class_id, color in color_map.items():
        seg_img[mask_resized == class_id, :3] = color
        seg_img[mask_resized == class_id, 3] = 255

    # Gerçek maskeyi görselleştirme
    gt_img = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        gt_img[gt_mask_resized == class_id] = color

    original_img = np.array(resized_image)
    blended = original_img.copy()
    for c in range(3):
        blended[:, :, c] = (alpha * seg_img[:, :, c] + (1 - alpha) * original_img[:, :, c]).astype(np.uint8)

    # Görselleştirme
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 3 subplot oluşturduk
    ax[0].imshow(resized_image)
    ax[0].set_title("Giriş Görseli (640x640)")
    ax[0].axis("off")

    ax[1].imshow(blended)
    ax[1].set_title("Segmentasyon Sonucu")
    ax[1].axis("off")

    ax[2].imshow(gt_img)
    ax[2].set_title("Gerçek Maskeyi Göster")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# Modeli yükle
model_path = "C:/Users/onurd/Desktop/archive/best_segformer_model01.pth"
model = load_trained_model(model_path)

# Test görseli
image_path = "C:/Users/onurd/Desktop/archive/val/images/c060ea1f-da217407.jpg"

# Segmentasyon yap, doğruluk hesapla ve göster
visualize_segmentation_and_accuracy(image_path, model)
