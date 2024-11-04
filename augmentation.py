import cv2
import numpy as np
import random
import os
from skimage.io import imsave

input_images_path = './resized_details/'
input_masks_path = '.\\annotation\\masks_categorical'

augmented_images_path = '.\\augmentation\\aug_details'
categorical_masks_path = '.\\augmentation\\aug_masks_categorical'
colour_masks_path = '.\\augmentation\\aug_masks_colour'

os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs(categorical_masks_path, exist_ok=True)
os.makedirs(colour_masks_path, exist_ok=True)

# случайный поворот
def random_flip(image, mask):
    if random.choice([True, False]):
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    if random.choice([True, False]):
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask

# случайный кроп
def random_crop(image, mask, crop_size=(800, 800)):  
    h, w = image.shape[:2]
    ch, cw = crop_size
    x = random.randint(0, max(1, w - cw))
    y = random.randint(0, max(1, h - ch))
    image = image[y:y + ch, x:x + cw]
    mask = mask[y:y + ch, x:x + cw]
    return image, mask


# случайное изменение яркости
def random_brightness(image):
    factor = 0.8 + random.uniform(0, 0.7)  
    return np.clip(image * factor, 0, 255).astype(np.uint8)

# случайный поворота на угол
def random_rotate(image, mask):
    angle = random.choice([90, 180, 270])
    image = cv2.rotate(image, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
    mask = cv2.rotate(mask, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
    return image, mask

# преобразование категориальных значений маски в цветное изображение
def logits2rgb(img):
    colours = [
        [0, 0, 0],        # черный (фон)
        [187, 207, 74],   # зеленый
        [0, 108, 132],    # синий
        [255, 204, 184],  # желтый
        [200, 0, 10],     # красный
        [226, 232, 228],  # белый
        [232, 167, 53]    # оранжевый
    ]

    h, w = img.shape
    col = np.zeros((h, w, 3), dtype=np.uint8)
    for val in np.unique(img):
        mask = (img == val)
        col[mask] = colours[int(val)]
    return col

# аугментация и сохранения
def augment_and_save(input_images_path, input_masks_path):

    input_images = sorted([os.path.join(input_images_path, f) for f in os.listdir(input_images_path) if f.endswith('.png')])
    input_masks = sorted([os.path.join(input_masks_path, f) for f in os.listdir(input_masks_path) if f.endswith('.png')])

    for i, (img_path, mask_path) in enumerate(zip(input_images, input_masks)):

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # сохраняем оригиналы изображения и маски
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        imsave(os.path.join(augmented_images_path, f"{base_name}.png"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imsave(os.path.join(categorical_masks_path, f"{base_name}-mask-cat.png"), mask)
        mask_colour = logits2rgb(mask)
        imsave(os.path.join(colour_masks_path, f"{base_name}-mask-col.png"), mask_colour)

        # применяем аугментации
        for aug_num in range(1, 4):
            if aug_num == 1:
                # случайный флип и кроп
                image_aug, mask_aug = random_flip(image, mask)
                image_aug, mask_aug = random_crop(image_aug, mask_aug)
            elif aug_num == 2:
                # случайное изменение яркости
                image_aug = random_brightness(image)
                mask_aug = mask
            elif aug_num == 3:
                # случайный поворот на 90, 180 или 270 градусов
                image_aug, mask_aug = random_rotate(image, mask)

            # сохраняем аугментированные изображения
            aug_img_name = f"{base_name}-aug_{aug_num}.png"
            imsave(os.path.join(augmented_images_path, aug_img_name), cv2.cvtColor(image_aug, cv2.COLOR_BGR2RGB))

            categorical_mask_name = f"{base_name}-aug_{aug_num}-mask-cat.png"
            imsave(os.path.join(categorical_masks_path, categorical_mask_name), mask_aug)

            colour_mask_name = f"{base_name}-aug_{aug_num}-mask-col.png"
            mask_colour_aug = logits2rgb(mask_aug)
            imsave(os.path.join(colour_masks_path, colour_mask_name), mask_colour_aug)

augment_and_save(input_images_path, input_masks_path)
