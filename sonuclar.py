import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import os

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

# Görseli dönüştürme (512x512 boyutuna getirerek)
def preprocess_image(image_path, img_size=(640,640)):
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize(img_size, Image.BILINEAR)
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image_resized).unsqueeze(0), image_resized

# Ground truth maskeleri yükle ve birleştir
def load_combined_ground_truth_mask(image_filename, img_size=(640,640)):
    image_id = os.path.splitext(image_filename)[0]
    base_path = "C:/Users/onurd/Desktop/archive/val/masks"

    drivable_path = os.path.join(base_path, f"{image_id}_drivable_mask.png")
    lane_path = os.path.join(base_path, f"{image_id}_lane_mask.png")

    if not os.path.exists(drivable_path) or not os.path.exists(lane_path):
        print(f"Maske dosyaları bulunamadı: {image_filename}")
        return None

    drivable_mask = Image.open(drivable_path).convert("L").resize(img_size, Image.NEAREST)
    lane_mask = Image.open(lane_path).convert("L").resize(img_size, Image.NEAREST)

    drivable_np = np.array(drivable_mask)
    lane_np = np.array(lane_mask)

    combined_mask = np.zeros_like(drivable_np)
    combined_mask[drivable_np > 0] = 1
    combined_mask[lane_np > 0] = 2  # lane overwrite yapar

    return combined_mask

# Metrics hesaplama fonksiyonu
def calculate_metrics_per_class(pred_mask, gt_mask, class_id):
    mask = (gt_mask == class_id) | (pred_mask == class_id)  # ilgili sınıfın olduğu pikseller
    pred = pred_mask[mask] == class_id
    gt = gt_mask[mask] == class_id

    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return precision, recall, iou

# Doğruluk hesaplama fonksiyonu
def calculate_accuracy(pred_mask, gt_mask, class_id):
    mask = gt_mask == class_id
    total = np.sum(mask)
    if total == 0:
        return 0
    correct = np.sum((pred_mask == class_id) & mask)
    accuracy = correct / total
    return accuracy

# Model ile segmentasyon ve ölçüm yap
def evaluate_on_dataset(model, images_dir, max_images=5000, img_size=(640,640)):
    image_files = sorted(os.listdir(images_dir))[:max_images]

    total_drivable_iou = 0
    total_lane_iou = 0

    total_drivable_acc = 0
    total_lane_acc = 0

    count = 0

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)

        # Görsel ön işlem
        image_tensor, resized_image = preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(pixel_values=image_tensor).logits

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        pred_mask_resized = np.array(Image.fromarray(pred_mask.astype(np.uint8)).resize(resized_image.size, Image.NEAREST))

        # GT maskesi yükle
        gt_mask = load_combined_ground_truth_mask(image_file, img_size)
        if gt_mask is None:
            continue  # GT maskesi yoksa atla

        # Drivable IoU, Precision, Recall
        _, _, drivable_iou = calculate_metrics_per_class(pred_mask_resized, gt_mask, class_id=1)
        drivable_acc = calculate_accuracy(pred_mask_resized, gt_mask, class_id=1)

        # Lane IoU, Precision, Recall
        _, _, lane_iou = calculate_metrics_per_class(pred_mask_resized, gt_mask, class_id=2)
        lane_acc = calculate_accuracy(pred_mask_resized, gt_mask, class_id=2)

        total_drivable_iou += drivable_iou
        total_lane_iou += lane_iou
        total_drivable_acc += drivable_acc
        total_lane_acc += lane_acc

        count += 1

        if count % 100 == 0:
            print(f"{count} görsel işlendi...")

    # Ortalama hesaplama
    avg_drivable_iou = total_drivable_iou / count if count > 0 else 0
    avg_lane_iou = total_lane_iou / count if count > 0 else 0
    avg_drivable_acc = total_drivable_acc / count if count > 0 else 0
    avg_lane_acc = total_lane_acc / count if count > 0 else 0
    avg_acc = (avg_drivable_acc + avg_lane_acc) / 2

    print("\n--- 5000 Fotoğraf Sonrası Ortalama Değerlendirme ---")
    print(f"Drivable IoU (Sınıf 1): {avg_drivable_iou * 100:.2f}%")
    print(f"Lane IoU (Sınıf 2): {avg_lane_iou * 100:.2f}%")
    print(f"Drivable Doğruluk (Sınıf 1): {avg_drivable_acc * 100:.2f}%")
    print(f"Lane Doğruluk (Sınıf 2): {avg_lane_acc * 100:.2f}%")
    print(f"İki Sınıfın Ortalama Doğruluğu: {avg_acc * 100:.2f}%")

# Modeli yükle
model_path = "C:/Users/onurd/Desktop/archive/best_segformer_model.pth"
model = load_trained_model(model_path)

# Değerlendir
images_dir = "C:/Users/onurd/Desktop/archive/val/images"
evaluate_on_dataset(model, images_dir, max_images=5000)
