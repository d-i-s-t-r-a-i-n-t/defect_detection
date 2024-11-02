import os 
import numpy as np
import random
from torchvision import transforms
from PIL import Image

# путь к папке с изображениями и масками
images_dir = './details_resize/' 
masks_dir = './decoded_masks/' # (попробовать на черном/мб даже не разъединять по классам/мб из категориальных)
output_images_dir = './aug_img/'
output_masks_dir = './aug_masks/'

# создаем выходные директории, если они не существуют
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# получаем список всех изображений
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# определяем класс для аугментации
class ImageMaskTransform:
    def __init__(self):
        self.flip_horizontal = transforms.RandomHorizontalFlip(p=1)
        self.flip_vertical = transforms.RandomVerticalFlip(p=1)
        self.rotation = transforms.RandomRotation(degrees=25)
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
        self.crop = transforms.RandomCrop(size=(800, 800))

    def __call__(self, image, mask):
        # применяем горизонтальное отражение
        if random.random() < 0.5:
            image = self.flip_horizontal(image)
            mask = self.flip_horizontal(mask)

        # применяем вертикальное отражение
        if random.random() < 0.5:
            image = self.flip_vertical(image)
            mask = self.flip_vertical(mask)

        # поворот
        rotation_angle = random.uniform(-25, 25)
        image = transforms.functional.rotate(image, rotation_angle)
        mask = transforms.functional.rotate(mask, rotation_angle)

        # обрезание (важно, чтобы обрезка применялась к одинаковым координатам)
        image, mask = self.crop_to_same_area(image, mask)

        return image, mask

    def crop_to_same_area(self, image, mask):
        # получаем размер обрезаемого изображения
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(800, 800))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        return image, mask

# функция для аугментации изображения и маски
def augment_image_and_mask(image_file):
    # загружаем изображение
    image_path = os.path.join(images_dir, image_file)
    image = Image.open(image_path).convert("RGB")  # загружаем изображение

    # ищем соответствующие маски
    base_name = image_file.split('.')[0]  # получаем базовое имя без расширения
    mask_files = [f for f in os.listdir(masks_dir) if f.startswith(f'decoded_{base_name}') and f.endswith('.png')]
    
    transform = ImageMaskTransform()

    # аугментация для каждой маски
    for i in range(30):  # создаем 30 аугментированных вариантов
        for mask_file in mask_files:
            if mask_file.split("_")[1].split(".")[0] == base_name:
                mask_path = os.path.join(masks_dir, mask_file)
                mask = Image.open(mask_path).convert("L")  # загружаем маску в градациях серого

                # применяем аугментацию
                augmented_image, augmented_mask = transform(image, mask)

                # сохраняем аугментированное изображение
                augmented_image_filename = os.path.join(output_images_dir, f'{base_name}_aug_{i}-mask.png_{mask_file.split("_")[-1]}')
                augmented_image.save(augmented_image_filename)

                # сохраняем аугментированную маску
                augmented_mask_filename = os.path.join(output_masks_dir, f'{base_name}_aug_{i}-mask.png_{mask_file.split("_")[-1]}')
                augmented_mask.save(augmented_mask_filename)

# последовательная обработка изображений
for image_file in image_files:
    augment_image_and_mask(image_file)

print("Аугментация завершена!")
