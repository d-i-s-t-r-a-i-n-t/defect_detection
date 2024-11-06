import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Папки для исходных и сохраненных изображений
input_folder = r'.\details'
output_folder = r'.\processed_details'

# Создаем папку для сохранения результатов, если она не существует
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Получаем список всех файлов в папке
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# Обработка каждого изображения
for image_file in image_files:
    # Загрузка изображения
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Преобразование в цветовое пространство LAB для улучшения выделения объектов и уменьшения влияния теней
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Выравнивание гистограммы (для уменьшения влияния теней)
    l, a, b = cv2.split(lab_image) # разделение на 3 канала
    l = cv2.equalizeHist(l)  # Выравнивание гистограммы компонента L для улучшения яркости
    lab_image = cv2.merge((l, a, b)) # объединение обратно

    # Преобразование обратно в цветовое пространство BGR
    image_equalized = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Применение фильтра Гаусса для сглаживания изображения и уменьшения шума (включая тени)
    image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

    # Применение LBP для выделения текстурных признаков (можно использовать радиус 1 и круг 8 точек)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY), n_points, radius, method='uniform')

    # Преобразование LBP изображения в формат uint8
    lbp_image = np.uint8(lbp_image)

    # Сегментация изображения с использованием GrabCut
    mask = np.zeros(image_blurred.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (10, 10, image_blurred.shape[1]-10, image_blurred.shape[0]-10)
    cv2.grabCut(image_blurred, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    # Сглаживание контуров с помощью морфологических операций
    kernel = np.ones((5, 5), np.uint8)  # Ядро для морфологических операций
    result_smooth = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)  # Закрытие (dilation + erosion)
    result_smooth = cv2.morphologyEx(result_smooth, cv2.MORPH_OPEN, kernel)  # Открытие (erosion + dilation)

    # Применение дополнительного сглаживания через гауссов фильтр
    result_smooth = cv2.GaussianBlur(result_smooth, (5, 5), 0)
    # Сохранение обработанного изображения
    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, result)

      

    print(f"Processed and saved: {image_file}")