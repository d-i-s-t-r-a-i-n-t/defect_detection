#!/usr/bin/env python3
import yaml
import requests
import cv2
import numpy as np
import ndjson
import os

# преобразование категориальных значений маски в цветное изображение
def logits2rgb(img):
    colours = {
        0: [0, 0, 0],         # black
        1: [187, 207, 74],    # green
        2: [0, 108, 132],     # blue
        3: [255, 204, 184],   # yellow
        4: [200, 0, 10],      # red
        5: [226, 232, 228],   # white
        6: [232, 167, 53],    # orange
    }
    col_img = np.zeros((*img.shape, 3), dtype=int)
    for val, color in colours.items():
        col_img[img == val] = color # пиксели с номером класса раскрашиваем в соотв. цвет
    return col_img

# загрузка масок, преобразование и сохранение в PNG
def get_mask(PROJECT_ID, api_key, colour, class_indices, dest_path_colour, dest_path_cat, log_file_path):
    os.makedirs(dest_path_cat, exist_ok=True)
    os.makedirs(dest_path_colour, exist_ok=True)

    with open('.\\annotation\\labelbox_segm.ndjson') as f, open(log_file_path, 'w') as log_file:
        data = ndjson.load(f)

        headers = {'Authorization': api_key}
        for item in data:
            base_name = item['data_row']['external_id'].replace(".JPG", "")
            # создаем маску с нулями по нужному размеру 
            mask_full = np.zeros((item['media_attributes']['height'], item['media_attributes']['width']))

            # проходимся по аннотированным объектам
            for obj in item['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']:
                class_idx = class_indices.get(obj['name'])   # получаем индекс класса объекта
                if class_idx is None:
                    continue
                
                
                with requests.get(obj['mask']['url'], headers=headers, stream=True) as r:
                    r.raw.decode_content = True
                    # декодируем загруженное изображение маски в черно-белом формате (grayscale)
                    mask_img = np.asarray(bytearray(r.raw.read()), dtype="uint8")
                    # находим белые пиксели объекта
                    mask = cv2.imdecode(mask_img, cv2.IMREAD_GRAYSCALE) == 255
                    mask_full[mask] = class_idx # устанавливаем для них значение - номер класса

            unique_classes = np.unique(mask_full)
            # вывод классов для каждой картинки и маски
            log_message = f"{base_name}: классы в маске - {unique_classes}"
            print(log_message)  # в консоль
            log_file.write(log_message + '\n')  # в лог-файл

            # сохраняем категориальное изображение маски
            cv2.imwrite(os.path.join(dest_path_cat, f"{base_name}-mask-cat.png"), mask_full)

            # сохраняем цветное изображение маски
            if colour and len(unique_classes) > 1:
                mask_col_img = cv2.cvtColor(logits2rgb(mask_full).astype('float32'), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(dest_path_colour, f"{base_name}-mask-col.png"), mask_col_img)

# загрузка конфигурации из файла config.yaml для авторизации в Labelbox
with open('.\\annotation\\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

get_mask(
    PROJECT_ID=config['project_id'],
    api_key=config['api_key'],
    colour=True,
    class_indices={
        "skol": 1,
        "skos": 2,
        "vmyatina": 3,
        "sled ot instrumenta": 4,
        "isnos": 5,
        "carapina": 6,
    },
    dest_path_colour='.\\annotation\\masks_colour/',
    dest_path_cat='.\\annotation\\masks_categorical/',
    log_file_path='.\\annotation\\mask_classes_log.txt'
)
