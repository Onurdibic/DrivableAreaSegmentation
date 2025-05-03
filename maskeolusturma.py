import json
import numpy as np
import cv2
import os

# JSON dosyasını yükleme
json_path = r'C:\Users\onurd\Desktop\archive\output_coco_format.json'
with open(json_path, 'r') as f:
    data = json.load(f)

# Maskeleri kaydedeceğimiz dizin
output_dir = 'masks'
os.makedirs(output_dir, exist_ok=True)

# Görüntülerin bulunduğu dizin
images_dir = r'C:\Users\onurd\Desktop\archive\train\images'

# Görüntü ve segmentasyon verilerini işleme
for image_info in data['images']:
    image_id = image_info['id']
    image_name = image_info['file_name']
    width = image_info['width']
    height = image_info['height']
    
    # Görüntüyü yükleyin
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Görüntü bulunamadı: {image_path}")
        continue
    
    image = cv2.imread(image_path)
    
    # Maskeleri başlat (her biri sıfırdan başlar)
    drivable_mask = np.zeros((height, width), dtype=np.uint8)  # drivable area maskesi
    lane_mask = np.zeros((height, width), dtype=np.uint8)  # lane maskesi
    
    # Anotasyon verisini işleme
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            segmentation = annotation['segmentation']
            
            # Maskeyi oluşturma: Segmentation, çokgen şeklinde pikselleri işaret eder
            if category_id == 1:  # drivable area
                for polygon in segmentation:
                    points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(drivable_mask, [points], color=1)  # drivable maskesinde 1 olarak işaretle
            elif category_id == 2:  # lane
                for polygon in segmentation:
                    points = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(lane_mask, [points], color=1)  # lane maskesinde 1 olarak işaretle
    
    # Maskeleri kaydet
    drivable_mask_filename = os.path.join(output_dir, f'{image_name}_drivable_mask.png')
    lane_mask_filename = os.path.join(output_dir, f'{image_name}_lane_mask.png')
    
    cv2.imwrite(drivable_mask_filename, drivable_mask * 255)  # Görüntüde daha iyi görmek için 255 ile çarparız
    cv2.imwrite(lane_mask_filename, lane_mask * 255)  # Görüntüde daha iyi görmek için 255 ile çarparız
    
   
