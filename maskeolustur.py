import json
import os
import cv2
import numpy as np

input_file = "C:/Users/onurd/Desktop/archive/val/annotations/bdd100k_labels_images_val.json"
output_dir = "C:/Users/onurd/Desktop/archive/maskeler"
image_width = 1280
image_height = 720

# Maskeleri kaydetmek için klasör oluştur
os.makedirs(output_dir, exist_ok=True)

with open(input_file, 'r') as f:
    data = json.load(f)

for item in data:
    # İki ayrı siyah maske
    drivable_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    lane_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for label in item['labels']:
        category = label.get('category')

        if 'poly2d' in label:
            for poly in label['poly2d']:
                vertices = poly['vertices']
                pts = np.array(vertices, dtype=np.int32).reshape((-1, 1, 2))

                if category == 'drivable area':
                    cv2.fillPoly(drivable_mask, [pts], 255)
                elif category == 'lane':
                    cv2.fillPoly(lane_mask, [pts], 255)

    base_name = os.path.splitext(item['name'])[0]
    drivable_path = os.path.join(output_dir, f"{base_name}_drivable_mask.png")
    lane_path = os.path.join(output_dir, f"{base_name}_lane_mask.png")

    cv2.imwrite(drivable_path, drivable_mask)
    cv2.imwrite(lane_path, lane_mask)

print("Tüm maske görüntüleri başarıyla oluşturuldu.")
