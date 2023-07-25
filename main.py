
import os
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Путь к папке с фотографиями
folder_path = "Graphiks"
# Получение списка файлов в папке
file_list = os.listdir(folder_path)

# Создание пустых массивов для хранения изображений и меток
images = []
labels = []

# Проход по каждому файлу
for file_name in file_list:
    # Полный путь к файлу
    file_path = os.path.join(folder_path, file_name)

    # Открытие изображения с помощью OpenCV
    img = cv2.imread(file_path)

    # Преобразование изображения в градацию серого цвета
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Изменение разрешения изображения на 64x64 с помощью Pillow
    resized_img = Image.fromarray(gray_img).resize((256,256))

    # Преобразование изображения в массив numpy
    img_array = np.array(resized_img)

    # Добавление изображения и метки в соответствующие массивы
    images.append(img_array)
    labels.append(3)  # Замените 0 на фактическую метку для данного изображения

# Преобразование массивов в формат, подходящий для обучения модели
images = np.array(images)
labels = np.array(labels)

# Создание модели сверточной нейронной сети
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(images, labels, epochs=20, batch_size=64)
model.save("new_modelka")