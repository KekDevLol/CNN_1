import tensorflow as tf
from keras.preprocessing import image
import numpy as np

# Загрузка обученной модели
model = tf.keras.models.load_model('new_model.h5')

# Определение функции для предобработки изображения
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Нормализация изображения
    return img_array

# Загрузка и предобработка входного изображения
input_image = 'Check/515.png'
preprocessed_image = preprocess_image(input_image)

# Получение предсказаний для предобработанного изображения
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions, axis=1)[0]

# Определение классов
class_labels = ['chertezh', 'graphiks', 'other']

# Вывод предсказанного класса
print(f"Предсказанный класс: {class_labels[predicted_class]}")