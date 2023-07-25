import os

# Путь к папке, в которой находятся изображения
folder_path = 'validation'

# Рекурсивная функция для удаления изображений без "rgb" в названии
def delete_images_without_rgb(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                if 'rgb_rgb_rgb' not in file.lower():
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

# Вызов функции для удаления изображений
delete_images_without_rgb(folder_path)