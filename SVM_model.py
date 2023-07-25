import os
import pickle

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize

# Путь к папке с данными
data_dir = 'train'

# Список классов
class_names = ['chertezh', 'graphiks', 'other']

# Загрузка и предобработка данных
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
        # Добавление изображения и метки в списки
        data.append(image.flatten())  # Преобразование изображения в одномерный вектор
        labels.append(i)

# Преобразование списков в массивы numpy
data = np.array(data)
labels = np.array(labels)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Создание и обучение SVM модели
model = svm.SVC(kernel='linear')

# Обучение модели с выводом прогресса
model.fit(X_train, y_train)

# Предсказание классов для тестовой выборки
y_pred = model.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность: ", accuracy)

# Оценка полноты модели
precision = precision_score(y_test, y_pred, average='weighted')
print("Полнота: ", precision)

# Оценка F-меры модели
f1 = f1_score(y_test, y_pred, average='weighted')
print("F-мера: ", f1)

# Сохранение модели в отдельный файл
model_filename = 'svm_model.pkl'
pickle.dump(model, open(model_filename, 'wb'))
print("Модель сохранена в файл: ", model_filename)
