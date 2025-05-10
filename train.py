import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanilan Cihaz: {device}")

# Parametreler
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 13
IMG_SIZE = (640, 640)

# Veri yollari
TRAIN_ANNOTATIONS_FILE = "C:/Users/onurd/Desktop/archive/train/annotations/bdd100k_labels_images_train.json"
VAL_ANNOTATIONS_FILE = "C:/Users/onurd/Desktop/archive/val/annotations/bdd100k_labels_images_val.json"
TRAIN_IMG_DIR = "C:/Users/onurd/Desktop/archive/train/images"
VAL_IMG_DIR = "C:/Users/onurd/Desktop/archive/val/images"

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=0.35, weight_ce=0.65, smooth=1.0):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss_ce = self.ce(inputs, targets)
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        loss_dice = 1 - dice_score.mean()
        return self.weight_dice * loss_dice + self.weight_ce * loss_ce

class BDD100KDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, subset_size=6000, mode="train"):
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.data = self.data[:subset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]['name']
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.mode == "train":
            drivable_mask_path = img_path.replace("train/images", "train/masks").replace(".jpg", ".jpg_drivable_mask.png")
            lane_mask_path = img_path.replace("train/images", "train/masks").replace(".jpg", ".jpg_lane_mask.png")
        else:
            drivable_mask_path = img_path.replace("val/images", "val/masks").replace(".jpg", "_drivable_mask.png")
            lane_mask_path = img_path.replace("val/images", "val/masks").replace(".jpg", "_lane_mask.png")
        drivable_mask = Image.open(drivable_mask_path).convert("L").resize(IMG_SIZE)
        lane_mask = Image.open(lane_mask_path).convert("L").resize(IMG_SIZE)
        drivable_mask = torch.from_numpy(np.array(drivable_mask))
        lane_mask = torch.from_numpy(np.array(lane_mask))
        combined_mask = torch.zeros_like(drivable_mask)
        combined_mask[drivable_mask > 0] = 1
        combined_mask[lane_mask > 0] = 2
        if self.transform:
            img = self.transform(img)
        return img, combined_mask.long()

def get_data_loaders():
    train_dataset = BDD100KDataset(TRAIN_ANNOTATIONS_FILE, TRAIN_IMG_DIR, transform, subset_size=6000, mode="train")
    val_dataset = BDD100KDataset(VAL_ANNOTATIONS_FILE, VAL_IMG_DIR, transform, subset_size=1500, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def load_model():
    config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config.num_labels = 3  # Sinif Sayisi belirlenir

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        config=config,
        ignore_mismatched_sizes=True
    )
    
    return model.to(device)

def evaluate_model(model, val_loader, criterion, num_classes=3):
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    iou_per_class = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += torch.sum(preds == masks).item()
            total_pixels += masks.numel()
            for cls in range(num_classes):
                pred_inds = (preds == cls)
                target_inds = (masks == cls)
                intersection = (pred_inds & target_inds).sum().item()
                union = (pred_inds | target_inds).sum().item()
                if union > 0:
                    iou = intersection / union
                    iou_per_class[cls].append(iou)
    avg_val_loss = total_loss / len(val_loader)
    pixel_accuracy = correct_pixels / total_pixels
    mean_iou = np.mean([np.mean(iou_per_class[cls]) if iou_per_class[cls] else 0 for cls in range(num_classes)])
    class_iou_avg = {cls: np.mean(iou_per_class[cls]) if iou_per_class[cls] else 0 for cls in iou_per_class}
    return avg_val_loss, pixel_accuracy, mean_iou, class_iou_avg

def visualize_predictions(model, val_loader):
    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)
            predictions = torch.argmax(outputs, dim=1)
            img_np = images[0].cpu().permute(1, 2, 0).numpy()
            mask_np = masks[0].cpu().numpy()
            pred_np = predictions[0].cpu().numpy()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_np)
            axs[0].set_title("Giriş Görseli")
            axs[0].axis("off")
            axs[1].imshow(mask_np, cmap="gray")
            axs[1].set_title("Gerçek Maske")
            axs[1].axis("off")
            axs[2].imshow(pred_np, cmap="gray")
            axs[2].set_title("Tahmin Maskesi")
            axs[2].axis("off")
            plt.tight_layout()
            plt.show()
            break

def train_model(model, train_loader, val_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = CombinedLoss(weight_dice=0.5, weight_ce=0.5)
    best_val_loss = float("inf")
    model.train()
    total_images = len(train_loader.dataset)
    processed_images = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            processed_images += images.size(0)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], Islenen: {processed_images}/{total_images*NUM_EPOCHS}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n Epoch [{epoch+1}/{NUM_EPOCHS}], Egitim Kayip: {avg_loss:.4f}")
        
        # Görselleştirme her epoch sonrası
        #visualize_predictions(model, val_loader)

        val_loss, accuracy, mean_iou, class_iou_avg = evaluate_model(model, val_loader, criterion)
        print(f"Epoch [{epoch+1}], Dogrulama Kayip: {val_loss:.4f}, Genel Dogruluk: {accuracy:.4f}, IoU: {class_iou_avg}")
        print(f"Drivable Alan IoU: {class_iou_avg[1]*100:.2f}%")
        print(f"Lane IoU: {class_iou_avg[2]*100:.2f}%\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_segformer_model02.pth")
            print("En iyi model guncellendi ve kaydedildi\n")

    print("Egitim tamamlandi")



def main():
    train_loader, val_loader = get_data_loaders()
    model = load_model()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
