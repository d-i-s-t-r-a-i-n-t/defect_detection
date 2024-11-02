import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models
from tensorflow.python.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import time
from tensorflow.python.keras.engine import data_adapter

# фикс для использования DataAdapter в новых версиях TensorFlow
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

images_dir = './aug_img/'
masks_dir = './aug_masks/'

IMG_HEIGHT, IMG_WIDTH = 256, 256
num_classes = len(class_labels)+1  # количество классов, включая фон

# загрузка и предобработка изображений и масок
def load_data(images_dir, masks_dir, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    images = []
    masks = []
    
    for mask_file in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask_file)
        
        # грузим маску
        mask = Image.open(mask_path).resize(img_size)
        mask = np.array(mask)  # Преобразуем в массив numpy без нормализации

        # one-hot кодирование маски (еще проверить)
        one_hot_mask = np.zeros((*mask.shape, num_classes), dtype=np.float32)
        for i in range(num_classes):
            one_hot_mask[..., i] = np.where(mask == i, 1.0, 0.0)
        
        masks.append(one_hot_mask)
        
        # грузим изображение
        image_file = mask_file  # имена в папках совпадают
        image_path = os.path.join(images_dir, image_file)
        image = Image.open(image_path).resize(img_size)
        image = np.array(image) / 255.0  # нормализация
        images.append(image)

    return np.array(images), np.array(masks)

# загрузка данных
images, masks = load_data(images_dir, masks_dir)

# разделение данных на тренировочный, валидационный и тестовый наборы (проверить разделение)
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_masks, test_size=0.25, random_state=42)

# преобразование в TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks)).batch(8)
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks)).batch(8)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks)).batch(8)

# создание модели U-Net (посмотреть настройки)
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3)):
    inputs = layers.Input(input_size)
    
    # кодировщик
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # базовый слой
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    
    # декодер
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
        print(f" - loss: {logs['loss']:.4f} - mean_io_u: {logs['mean_io_u']:.4f} - val_loss: {logs['val_loss']:.4f} - val_mean_io_u: {logs['val_mean_io_u']:.4f}")

# создание и компиляция модели (посмотреть настройки)
model = unet_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=num_classes)])

# добавление преждевременной остановки
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
progress = TrainingProgress()

# обучение модели
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    callbacks=[early_stopping, progress]
)

# оценка модели на тестовом наборе 
test_loss, test_mean_io_u = model.evaluate(test_dataset)
print(f"Тестовый MeanIoU: {test_mean_io_u}") # (посмотреть еще метрики)

# cохранение модели (сохр. лучший рез)
model.save("unet_model.h5")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_io_u'], label='Training MeanIoU')
plt.plot(history.history['val_mean_io_u'], label='Validation MeanIoU')
plt.xlabel('Epoch')
plt.ylabel('MeanIoU')
plt.legend()
plt.title('Training and Validation MeanIoU')

plt.show()
