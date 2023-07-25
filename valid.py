import os
import cv2
from tqdm import tqdm

image_directory = 'validation'
target_size = (256, 256)

# Проход по папкам и валидация изображений
for root, dirs, files in os.walk(image_directory):
    for file in tqdm(files):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Загрузка изображения в сером цвете
            resized_image = cv2.resize(image, target_size)  # Изменение размера изображения

            # Сохранение измененного изображения
            cv2.imwrite(image_path, resized_image)

print("Валидация и преобразование изображений завершены.")