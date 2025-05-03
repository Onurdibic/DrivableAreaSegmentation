import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# Hiperparametreler
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
IMG_SIZE = (512, 1024)  


ANNOTATIONS_FILE = "C:/Users/onurd/Desktop/archive/train/annotations/bdd100k_labels_images_train.json"
IMG_DIR = "C:/Users/onurd/Desktop/archive/train/images"


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])


class BDD100KDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, subset_size=1000):
        self.img_dir = img_dir
        self.transform = transform


        with open(annotations_file, 'r') as f:
            self.data = json.load(f)


        self.data = self.data[:subset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx]['name'])
        img = Image.open(img_path).convert("RGB")


        drivable_mask_path = img_path.replace("images", "masks").replace(".jpg", ".jpg_drivable_mask.png")
        lane_mask_path = img_path.replace("images", "masks").replace(".jpg", ".jpg_lane_mask.png")

        drivable_mask = Image.open(drivable_mask_path).convert("L")
        lane_mask = Image.open(lane_mask_path).convert("L")


        drivable_mask = torch.from_numpy(np.array(drivable_mask))
        lane_mask = torch.from_numpy(np.array(lane_mask))


        combined_mask = torch.zeros_like(drivable_mask)
        combined_mask[drivable_mask > 0] = 1
        combined_mask[lane_mask > 0] = 2

        # Dönüşümleri uygula
        if self.transform:
            img = self.transform(img)

        # Maskeyi long formatında döndür
        mask = combined_mask.long()
        return img, mask


def get_data_loaders():
    train_dataset = BDD100KDataset(ANNOTATIONS_FILE, IMG_DIR, transform, subset_size=1000)  # İlk 5000 görüntüyü al
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def load_model():
    config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=3)
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", config=config)

    model.config.num_labels = 3  # 3 sınıf: arka plan, sürülebilir alan, şerit
    model = model.to(device)
  
    
    return model


def train_model(model, train_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    total_images = len(train_loader.dataset)  # Toplam görsel sayısı
    processed_images = 0  # İşlenen görsel sayısı

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits

            # Çıktıyı maskeye uyumlu hale getirme
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode="bilinear", align_corners=False)

            # Kayıp hesaplama ve geri yayılım
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            processed_images += images.size(0)  # İşlenen görsel sayısını artır


            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], İşlenen görsel sayısı: {processed_images}/{total_images*10}")


            torch.cuda.empty_cache()


        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")


    torch.save(model.state_dict(), "segformer_bdd100ke.pth")
    print("Model eğitimi tamamlandı ve kaydedildi!")



def main():
    train_loader = get_data_loaders()
    model = load_model()  # Modeli bir kez yükle
    train_model(model, train_loader)

if __name__ == "__main__":
    main()
