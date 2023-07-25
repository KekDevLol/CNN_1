import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Загрузка  модели
model = tf.keras.models.load_model('vit_model.h5')

# Создание генератора
test_datagen = ImageDataGenerator(rescale=1./255)

# Загрузка тестовых данных
test_generator = test_datagen.flow_from_directory(
    directory='validation',
    target_size=(256, 256),
    color_mode='rgb',
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Получение предсказаний модели для тестовых данных
predictions = model.predict(test_generator)
predicted_classes = tf.argmax(predictions, axis=1).numpy()
true_classes = test_generator.classes

# Вычисление метрик производительности
accuracy = accuracy_score(true_classes, predicted_classes)
precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')
auc_roc = roc_auc_score(true_classes, predictions, average='weighted', multi_class='ovr')

# Вывод метрик
print(f"Точность: {accuracy}")
print(f"Полнота: {recall}")
print(f"F-мера: {f1}")
print(f"AUC-ROC: {auc_roc}")