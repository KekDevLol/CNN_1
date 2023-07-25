import os
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# Путь к папке с данными
data_dir = 'train'

# Список классов
class_names = ['chertezh', 'graphiks', 'other']

# Загрузка данных
data = []
labels = []

# Проход по каждому классу
for i, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    # Проход по каждому изображению в классе
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        # Загрузка изображения и изменение размера
        image = imread(image_path, as_gray=True)
        image = resize(image, (256, 256))
        # Извлечение HOG признаков
        features = hog(image, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
        # Добавление признаков и метки в списки
        data.append(features)
        labels.append(i)

# Преобразование списков в массивы numpy
data = np.array(data)
labels = np.array(labels)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Создание и обучение модели наивного байеса
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозирование классов для тестовой выборки
y_pred = nb_model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность:", accuracy)

# Оценка полноты модели
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
print("Точность по классам:", precision)

# Оценка F-меры модели
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1-мера:", f1)

# Сохранение модели в отдельный файл
model_name = 'nb_model.pkl'
pickle.dump(nb_model, open(model_name, 'wb'))
print("Модель сохранена в файл:", model_name)