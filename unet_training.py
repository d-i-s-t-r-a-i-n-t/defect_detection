import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time

# фиксация для использования DataAdapter в новых версиях TensorFlow
from tensorflow.python.keras.engine import data_adapter
def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)
data_adapter._is_distributed_dataset = _is_distributed_dataset

# маппинг классов
class_labels = { 
    1: "skol",
    2: "skos",
    3: "vmyatina",
    4: "sled_ot_instrumenta",
    5: "isnos",
    6: "carapina"
}

images_dir = '.\\augmentation\\aug_details'
masks_dir = '.\\augmentation\\aug_masks_categorical'

IMG_HEIGHT, IMG_WIDTH = 512, 512
num_classes = len(class_labels) + 1  # количество классов, включая фон

# загрузка и предобработка изображений и масок
def load_data(images_dir, masks_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    masks = []
    
    for mask_file in os.listdir(masks_dir):
        base_name = mask_file.split('-mask-cat')[0] + '.png'
        img_path = os.path.join(images_dir, base_name)
        if not os.path.exists(img_path):
            continue
        
        image = Image.open(img_path).resize(img_size)
        image = np.array(image) / 255.0  # нормализация
        images.append(image)

        mask_path = os.path.join(masks_dir, mask_file)
        mask = Image.open(mask_path).resize(img_size, Image.NEAREST)
        mask = np.array(mask, dtype=np.int32)

        one_hot_mask = np.eye(num_classes)[mask]
        masks.append(one_hot_mask)
    
    return np.array(images), np.array(masks)

# загрузка данных
images, masks = load_data(images_dir, masks_dir)

# разделение данных на тренировочный, валидационный и тестовый наборы
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.25, random_state=42)

# преобразование в TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(8)

# функция для расчета коэффициента Дайса
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# dice_loss для комбинирования
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# комбинированная функция потерь с categorical_crossentropy для стабильности обучения
def combined_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# создание модели U-Net
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = layers.Input(input_size)
    
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    u1 = layers.UpSampling2D((2, 2))(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    
    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)
    
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

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
    epochs=3,
    callbacks=[early_stopping, progress]
)

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
