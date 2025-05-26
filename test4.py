import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation
import os
from transformers import SegformerConfig
import cv2

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
def preprocess_image(image_path, img_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(img_size, Image.BILINEAR)

    transform = transforms.Compose([transforms.ToTensor()])

    return transform(image_resized).unsqueeze(0), image_resized

# Ground truth maskeleri yükle ve birleştir
def load_combined_ground_truth_mask(image_filename, img_size=(512,512)):
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

# Gelişmiş metrik hesaplama fonksiyonu
def calculate_metrics_per_class(pred_mask, gt_mask, class_id):
    # Sadece ilgili sınıfın olduğu veya tahmin edildiği pikselleri al
    mask = (gt_mask == class_id) | (pred_mask == class_id)

    pred = pred_mask == class_id
    gt = gt_mask == class_id

    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)
    TN = np.sum((gt_mask != class_id) & (pred_mask != class_id))  # Tüm görüntüde ilgili sınıf dışındaki piksellerin doğru negatifleri

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "accuracy": accuracy,
        "specificity": specificity,
        "f1": f1
    }


# Segmentasyon sonuçlarını renkli görselleştirip kaydet
def save_segmented_image(image_path, pred_mask_resized, gt_mask, alpha=0.5):
    image_filename = os.path.basename(image_path)
    image_id = os.path.splitext(image_filename)[0]
    image_tensor, resized_image = preprocess_image(image_path)
    width, height = resized_image.size
    original_img = np.array(resized_image)

    color_overlay = np.zeros((height, width, 3), dtype=np.uint8)

    # Kesişimler ve farklılıklar
    intersection_drivable = (pred_mask_resized == 1) & (gt_mask == 1)
    intersection_lane = (pred_mask_resized == 2) & (gt_mask == 2)
    green_only = (pred_mask_resized == 1) & (gt_mask != 1)
    red_only = (pred_mask_resized == 2) & (gt_mask != 2)

    # Renkler (BGR formatında cv2 için)
    # Burada RGB kullandık, cv2.cvtColor ile dönüşüm yapılacak
    color_overlay[intersection_drivable] = [0, 0, 255]      # Mavi: doğru drivable
    color_overlay[intersection_lane] = [255, 255, 0]       # Sarı: doğru lane
    color_overlay[green_only] = [0, 255, 0]                # Yeşil: sadece tahmin drivable
    color_overlay[red_only] = [255, 0, 0]                  # Kırmızı: sadece tahmin lane

    blended = ((1 - alpha) * original_img + alpha * color_overlay).astype(np.uint8)

    # Metrikleri hesapla
    metrics1 = calculate_metrics_per_class(pred_mask_resized, gt_mask, class_id=1)
    metrics2 = calculate_metrics_per_class(pred_mask_resized, gt_mask, class_id=2)

    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    metrics_text = [
        f"Drivable Area: Prec={metrics1['precision']*100:.1f}% Rec={metrics1['recall']*100:.1f}% IoU={metrics1['iou']*100:.1f}",
        f"% F1={metrics1['f1']*100:.1f}% Acc={metrics1['accuracy']*100:.1f}% Spec={metrics1['specificity']*100:.1f}%",
        f"Lane: Prec={metrics2['precision']*100:.1f}% Rec={metrics2['recall']*100:.1f}% IoU={metrics2['iou']*100:.1f}",
        f"% F1={metrics2['f1']*100:.1f}% Acc={metrics2['accuracy']*100:.1f}% Spec={metrics2['specificity']*100:.1f}%"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255)  # Beyaz

    for idx, line in enumerate(metrics_text):
        cv2.putText(blended_bgr, line, (10, 20 + idx * 20), font, font_scale, color, thickness, cv2.LINE_AA)

    output_dir = "C:/Users/onurd/Desktop/archive/segmentation_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_path = os.path.join(output_dir, f"{image_id}_segmented.png")
    cv2.imwrite(result_path, blended_bgr)
    print(f"Segmentasyon sonucu kaydedildi: {result_path}")

# Görselleştir ve kaydetme işlevi
def visualize_and_save_segmentation(image_path, model):
    image_tensor, resized_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(pixel_values=image_tensor).logits

    pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    pred_mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(resized_image.size, Image.NEAREST))
    image_filename = os.path.basename(image_path)
    gt_mask = load_combined_ground_truth_mask(image_filename)

    if gt_mask is None:
        return

    save_segmented_image(image_path, pred_mask_resized, gt_mask)

# Modeli yükle
model_path = "C:/Users/onurd/Desktop/archive/best_segformer_model03.pth"
model = load_trained_model(model_path)

# Fotoğrafları sırayla işleme
image_dir = "C:/Users/onurd/Desktop/archive/val/images"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()

# İlk 5000 fotoğrafı atlamak için sayaç
start_index = 5000

# Fotoğraf işleme döngüsü
for i, image_filename in enumerate(image_files[start_index:], start=start_index + 1):
    image_path = os.path.join(image_dir, image_filename)
    print(f"İşlenen Fotoğraf {i}: {image_filename}")
    visualize_and_save_segmentation(image_path, model)
