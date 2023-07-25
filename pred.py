import tensorflow as tf

model = tf.keras.models.load_model('new_model.h5')
image_directory = 'validation'
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
image_data = image_generator.flow_from_directory(
    image_directory,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)
predictions = model.predict(image_data)
predicted_classes = tf.argmax(predictions, axis=1).numpy()
class_labels = list(image_data.class_indices.keys())
true_labels = image_data.labels

# Вычисление точности
accuracy = sum(predicted_classes == true_labels) / len(true_labels)
print(f"Точность предсказаний: {accuracy * 100:.2f}%")

# Вывод предсказанных и истинных классов
for i, image_path in enumerate(image_data.filepaths):
    image_name = image_path.split('/')[-1]  # Имя изображения из пути
    predicted_class_label = class_labels[predicted_classes[i]]
    true_class_label = class_labels[true_labels[i]]
    print(f"Изображение: {image_name}, Предсказанный класс: {predicted_class_label}, Истинный класс: {true_class_label}")