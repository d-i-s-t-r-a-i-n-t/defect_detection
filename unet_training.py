import os
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras.src.callbacks import EarlyStopping, Callback
from keras import layers, models

from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time


# # фиксация для использования DataAdapter в новых версиях TensorFlow
# from keras.src.engine import data_adapter
# def _is_distributed_dataset(ds):
#     return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec) # проверка, является ли датасет распределенным
# data_adapter._is_distributed_dataset = _is_distributed_dataset # метод заново для того, чтобы не было проблем в совместимости с TF

# маппинг классов
class_labels = { 
    1: "skol",
    2: "skos",
    3: "vmyatina",
    4: "sled_ot_instrumenta",
    5: "isnos",
    6: "carapina"
}

images_dir = r'./augmentation/aug_details/' # путь к фото
masks_dir = r'./augmentation/aug_masks_categorical/' # путь к маскам

IMG_HEIGHT, IMG_WIDTH = 512, 512 # размер изоб-ий для модели
num_classes = len(class_labels) + 1  # количество классов, включая фон

# загрузка и предобработка изображений и масок
def load_data(images_dir, masks_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = [] # список для хранения изображени
    masks = [] # список для хранения масок
    
    for mask_file in os.listdir(masks_dir):  # цикл по всем маскам
        base_name = mask_file.split('-mask-cat')[0] + '.png' # имя изображения для маски
        img_path = os.path.join(images_dir, base_name)
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).resize(img_size) # открытие + ресайз изображения
        image = np.array(image) / 255.0  # нормализация
        images.append(image)

        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path).resize(img_size, Image.NEAREST) # открытие + ресайз маски
        mask = np.array(mask, dtype=np.int32) # преобразование маски в массив целых чисел

        one_hot_mask = np.eye(num_classes)[mask] # преобразование маски в one-hot
        masks.append(one_hot_mask)
    
    return np.array(images), np.array(masks)

# загрузка данных
images, masks = load_data(images_dir, masks_dir)

# разделение данных на тренировочный, валидационный и тестовый наборы
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42) # 80% на тренировку, 20% на тест
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.25, random_state=42) # 75% от тренировочного на обучение, 25% на валидацию

# преобразование в TensorFlow Dataset, батчи размером по 8
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(8)

# функция для расчета коэффициента Дайса (степень совпадения масок)
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, dtype=tf.float32))  # приведение к float32
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, dtype=tf.float32)) # преобразвание предсказанной маски в одномерный массив
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f) # рассчет кол-ва совпадений
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth) # рассчет самого коэф-та Дайса

# dice_loss для комбинирования
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# комбинированная функция потерь с categorical_crossentropy для стабильности обучения
def combined_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# создание модели U-Net
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    
    inputs = layers.Input(input_size)
    
    # Энкодер
    # Первый уровень
    c1 = layers.Conv2D(64, (3, 3), padding='same', dilation_rate=1)(inputs)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    c1 = layers.Conv2D(64, (3, 3), padding='same', dilation_rate=1)(c1)
    c1 = layers.LeakyReLU(alpha=0.1)(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Второй уровень
    c2 = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=2)(p1)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    c2 = layers.Conv2D(128, (3, 3), padding='same', dilation_rate=2)(c2)
    c2 = layers.LeakyReLU(alpha=0.1)(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Третий уровень
    c3 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=4)(p2)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)
    c3 = layers.Conv2D(256, (3, 3), padding='same', dilation_rate=4)(c3)
    c3 = layers.LeakyReLU(alpha=0.1)(c3)
    
    # Декодер
    # Первый уровень декодера с секретным соединением к c2
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])  # Секретное соединение
    c4 = layers.Conv2D(128, (3, 3), padding='same')(u1)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)
    c4 = layers.Conv2D(128, (3, 3), padding='same')(c4)
    c4 = layers.LeakyReLU(alpha=0.1)(c4)

    # Второй уровень декодера с секретным соединением к c1
    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])  # Секретное соединение
    c5 = layers.Conv2D(64, (3, 3), padding='same')(u2)
    c5 = layers.LeakyReLU(alpha=0.1)(c5)
    c5 = layers.Conv2D(64, (3, 3), padding='same')(c5)
    c5 = layers.LeakyReLU(alpha=0.1)(c5)

    # Выходной слой
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    
    # Создание модели
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model
    # inputs = layers.Input(input_size)
    
    # # блоки сверток и пулинга
    # c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    # c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    # p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    # c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    # p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    # c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # # блоки расширения с UpSampling
    # u1 = layers.UpSampling2D((2, 2))(c3)
    # u1 = layers.concatenate([u1, c2])
    # c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    # c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    # u2 = layers.UpSampling2D((2, 2))(c4)
    # u2 = layers.concatenate([u2, c1])
    # c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    # c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    # outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5) # выходной слой с классификацией по классам
    
    # model = models.Model(inputs=[inputs], outputs=[outputs])
    # return model

# колбэк для вывода прогресса обучения
class TrainingProgress(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"\nЭпоха {epoch + 1} начата.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        print(f"Эпоха {epoch + 1} завершена за {elapsed_time:.2f} секунд.")
        print(f" - loss: {logs['loss']:.4f} - val_loss: {logs['val_loss']:.4f}")

# создание и компиляция модели
model = unet_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=[dice_coefficient])

# добавление преждевременной остановки на основе val_dice_coefficient
early_stopping = EarlyStopping(monitor='val_dice_coefficient', mode='max', patience=5, restore_best_weights=True)
progress = TrainingProgress()

# обучение модели
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    callbacks=[early_stopping, progress]
)

model.summary()
model.save('model.keras')

# визуализация истории обучения
plt.figure(figsize=(12, 6))

# график функции потерь
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# график Dice Coefficient
plt.subplot(1, 2, 2)
plt.plot(history.history['dice_coefficient'], label='Training Dice Coefficient')
plt.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Training and Validation Dice Coefficient')

plt.show()
