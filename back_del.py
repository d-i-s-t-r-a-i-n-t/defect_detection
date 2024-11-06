# необходимо подкорректировать исходя из качества эталонных изображений

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

input_folder = r'.\details' # входные изображения
output_folder = r'.\processed_details' # выходные изображения

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Преобразование в цветовое пространство LAB для улучшения выделения объектов и уменьшения влияния теней
    '''
    LAB — это цветовое пространство, используемое для представления цвета, которое состоит из трех компонентов:
    L (Lightness) — яркость или светлота, варьирующаяся от черного (0) до белого (100). Она описывает количество света в изображении
    A (Green-Red) — цветовой компонент, который меняется от зеленого к красному
    B (Blue-Yellow) — цветовой компонент, который изменяется от синего к желтому
    '''
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # выравнивание гистограммы (для уменьшения влияния теней)
    l, a, b = cv2.split(lab_image) # разделение на 3 канала
    l = cv2.equalizeHist(l)  # выравнивание гистограммы компонента L для улучшения яркости
    lab_image = cv2.merge((l, a, b)) # объединение обратно

    # преобразование обратно в цветовое пространство BGR
    image_equalized = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # фильтр Гаусса для сглаживания изображения и уменьшения шума 
    '''
    Гауссов фильтр — это фильтр для размытия изображения, который используется для сглаживания и устранения шума
    Он применяет математическую функцию Гаусса для того, чтобы создать "размытое" изображение
    В зависимости от радиуса фильтра он может размыть изображение с разной степенью
    '''
    image_blurred = cv2.GaussianBlur(image_equalized, (5, 5), 0)

    # LBP для выделения текстурных признаков 
    '''
    Local Binary Pattern (LBP) — метод выделения текстурных признаков на изображении 
    Принцип работы LBP:
    Для каждого пикселя рассматриваются соседние пиксели в некотором радиусе
    Сравниваются значения интенсивности пикселя в центре и пикселей вокруг него
    Если значение интенсивности соседнего пикселя больше, чем у центрального, то этот пиксель кодируется единицей, иначе — нулем
    Полученное двоичное число интерпретируется как уникальный код, который описывает текстуру в этом местоположении
    LBP особенно полезен в выделении текстурных элементов, таких как края, линии и пятна
    '''
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(cv2.cvtColor(image_blurred, cv2.COLOR_BGR2GRAY), n_points, radius, method='uniform')

    # LBP изображение в формат uint8
    lbp_image = np.uint8(lbp_image)

    # сегментация изображения с GrabCut
    '''
    GrabCut — это метод сегментации изображений, который отделяет объекты на переднем плане от фона
    работает в два этапа:
    Инициализация прямоугольником: Устанавливается начальный прямоугольник, в котором должен находиться объект (у нас - почти все изображение)
    Алгоритм итеративного уточнения: Исходя из статистики цветовых моделей, он анализирует пиксели внутри и снаружи прямоугольника, 
    выделяя те, которые принадлежат фону, и те, которые могут быть частью объекта
    '''
    mask = np.zeros(image_blurred.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    rect = (10, 10, image_blurred.shape[1]-10, image_blurred.shape[0]-10)
    cv2.grabCut(image_blurred, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    # сглаживание контуров с помощью морфологических операций
    '''
    Эрозия: уменьшает область объекта на изображении, убирая "края". Она полезна для удаления мелких деталей, сглаживания контура
    Дилатация: увеличивает область объекта, добавляя пиксели к краям. Она заполняет "пробелы" на границах объектов
    Закрытие (closing): последовательная операция дилатации и затем эрозии. Закрытие помогает "сгладить" контуры объектов, убирая мелкие пробелы.
    Открытие (opening): операция эрозии, за которой следует дилатация. Применяется для удаления мелких объектов или шумов
    '''
    kernel = np.ones((5, 5), np.uint8)  # ядро для морфологических операций
    result_smooth = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)  # Закрытие 
    result_smooth = cv2.morphologyEx(result_smooth, cv2.MORPH_OPEN, kernel)  # Открытие 

    # дополнительное сглаживание через гауссов фильтр
    result_smooth = cv2.GaussianBlur(result_smooth, (5, 5), 0)

    output_image_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_image_path, result)

      

    print(f"result: {image_file}")
