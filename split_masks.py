import numpy as np
import cv2
import os

mask_folder = './masks_colour/'
output_masks_dir = './split_masks/'
os.makedirs(output_masks_dir, exist_ok=True)

# цвета классов
class_indices = {
    "skol": 1,
    "skos": 2,
    "vmyatina": 3,
    "sled ot instrumenta": 4,
    "isnos": 5,
    "carapina": 6
}

colours = {
    1: [74, 207, 187],
    2: [132, 108, 0],
    3: [184, 204, 255],
    4: [0, 0, 0],
    5: [228, 232, 226],
    6: [220, 214, 174]
}

background_color = [10, 0, 200]  

# преобразование маски с несколькими классами в несколько масок по классам
def split_mask_by_class(mask, mask_name):
    height, width = mask.shape[:2]
    for class_idx, color in colours.items():
        # находим пиксели, соответствующие текущему классу
        mask_indices = np.all(mask == color, axis=-1)
        
        # пропускаем сохранение, если в маске нет пикселей текущего класса
        if not np.any(mask_indices):
            continue
        
        class_mask = np.full((height, width, 3), background_color, dtype=np.uint8)
        class_mask[mask_indices] = color
        
        output_path = os.path.join(output_masks_dir, f"decoded_{mask_name}_{class_idx}.png")
        cv2.imwrite(output_path, class_mask)

for mask_name in os.listdir(mask_folder):
    mask_path = os.path.join(mask_folder, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is not None:
        split_mask_by_class(mask, mask_name.split(".png")[0])

print("Создание масок завершено. Результаты сохранены в папке:", output_masks_dir)
