#!/usr/bin/env python3
import yaml
import requests
import cv2
import numpy as np
import ndjson
import os

# преобразование категориальных значений маски в цветное изображение
def logits2rgb(img):
    # значения цветов для классов
    red = [200, 0, 10]
    green = [187, 207, 74]
    blue = [0, 108, 132]
    yellow = [255, 204, 184]
    black = [0, 0, 0]
    white = [226, 232, 228]
    cyan = [174, 214, 220]
    #orange = [232, 167, 53]

    colours = [red, green, blue, yellow, black, white, cyan] # orange 

    # создадим пустое цветное изображение по размерам маски
    shape = np.shape(img)
    h = int(shape[0])
    w = int(shape[1])
    col = np.zeros((h, w, 3))  
    unique = np.unique(img)  # получим уникальные значения классов маски

    for i, val in enumerate(unique):
        mask = np.where(img == val)  # определим область, принадлежащую классу val
        for j, row in enumerate(mask[0]):
            x = mask[0][j]
            y = mask[1][j]
            col[x, y, :] = colours[int(val)]  # присвоим пикселям цвет класса

    return col.astype(int) 

# загрузка масок, преобразование и сохранение в PNG
def get_mask(PROJECT_ID, api_key, colour, class_indices, destination_path_colour, destination_path_categorical):
    with open('.\\data_annotation\\labelbox_segm.ndjson') as f: # экспорт аннотаций из Labelbox
        data = ndjson.load(f)

        if not os.path.isdir(destination_path_categorical):
            os.mkdir(destination_path_categorical)
        if not os.path.isdir(destination_path_colour):
            os.mkdir(destination_path_colour)

        # проходимся по картинкам
        for i, d in enumerate(data):
            files_in_folder = os.listdir('./masks_categorical/')
            image_name = data[i]['data_row']['external_id']  
            label_name = image_name.replace(".JPG", "") + '-mask.png'  

            # проверяем, была ли создана уже такая маска
            if label_name not in files_in_folder:
                # создаем пустую маску с размерами исходного изображения
                mask_full = np.zeros((data[i]['media_attributes']['height'], data[i]['media_attributes']['width']))

                # проходимся по классам
                for idx, obj in enumerate(data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']):
                    # извлекаем имя и URL маски с аннотацией
                    name = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['name']
                    url = data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects'][idx]['mask']['url']

                    # получаем индекс класса из словаря 
                    cl = class_indices[name]
                    print(f'Класс {name} присвоен индексу класса {cl}')
                    
                    # загружаем маску из lb_segm.ndjson
                    headers = {'Authorization': api_key} # авторизируемся по ключу
                    with requests.get(url, headers=headers, stream=True) as r:
                        r.raw.decode_content = True
                        mask = r.raw
                        image = np.asarray(bytearray(mask.read()), dtype="uint8")
                        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)  # преобразуем изображение в градации серого

                    # назначаем пикселям изображения маски номер класса
                    mask = np.where(image == 255)
                    mask_full[mask] = cl

                # выводим уникальные значения классов в маске
                unique = np.unique(mask_full)
                print('Маски изображения содержат классы:')
                print(unique)

                # сохраняем цветное изображение маски
                if len(unique) > 1: 
                    if colour:
                        mask_full_colour = logits2rgb(mask_full)
                        mask_full_colour = cv2.cvtColor(mask_full_colour.astype('float32'), cv2.COLOR_RGB2BGR)
                    
                    cv2.imwrite(destination_path_colour + image_name.replace(".JPG", "") + '-mask.png', mask_full_colour)

                # сохраняем категориальное изображение маски
                cv2.imwrite(destination_path_categorical + image_name.replace(".JPG", "") + '-mask.png', mask_full)
            else:
                print(f'Файл {label_name} уже обработан!')

# загрузка конфигурации из файла config.yaml для авторизации в https://labelbox.com/
with open('.\\data_annotation\\config.yaml', 'r') as file:
    config = yaml.safe_load(file)

project_id = config['project_id']
api_key = config['api_key']
colour = True # False сохранит маску только в виде категориальных значений
destination_path_colour = './masks_colour/'
destination_path_categorical = './masks_categorical/'

# определим индексы для классов
class_indices = {
    "skol": 1,
    "skos": 2,
    "vmyatina": 3,
    "sled ot instrumenta": 4,
    "isnos": 5,
    "carapina": 6,
}

get_mask(project_id, api_key, colour, class_indices, destination_path_colour, destination_path_categorical)
