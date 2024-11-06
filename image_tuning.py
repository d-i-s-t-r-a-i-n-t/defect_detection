import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

input_folder = './details'
output_folder = './tuned_result'
file_name = "23.jpg"  

# улучшение контраста с CLAHE ( попробовать еще Gamma Correction)
# основная идея CLAHE заключается в выполнении выравнивания гистограммы локально, в меньших областях изображения, а не глобально
# CLAHE включает в себя два основных этапа:
# повышение контрастности
# ограничение контрастности - количество пикселей с очень высокой или очень низкой интенсивностью уменьшается
def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

# удаление шума (портит, где много шума, попробовать еще Bilateral Filter или Non-Local Means Denoising(используется среднее значение всех пикселей в изображении с учётом их сходства с целевым пикселем) и с/без)
def remove_noise(image):
    return cv2.medianBlur(image, 5)

# улучшение резкости (попробовать Unsharp Masking(усиливает локальный контраст изображения на тех участках, где изначально присутствовали резкие изменения градаций цвета) или High-Pass Filter)
def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    return cv2.filter2D(image, -1, kernel)

img_path = os.path.join(input_folder, file_name)
if os.path.exists(img_path):
    image = cv2.imread(img_path)

    original_image = image.copy()

    enhanced_image = enhance_contrast(image)
    denoised_image = remove_noise(enhanced_image)
    sharpened_image = sharpen_image(denoised_image)

    # отображение изображений в сетке
    images = [original_image, enhanced_image, denoised_image, sharpened_image]
    titles = ["Original", "Enhanced Contrast", "Denoised", "Sharpened"]

    plt.figure(figsize=(15, 8))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)  # 2 строки, 3 столбца
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, sharpened_image)
else:
    print(f"Файл {file_name} не найден в папке {input_folder}")

