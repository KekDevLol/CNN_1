import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.models import Sequential

# Создание генератора изображений
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory='train',
    target_size=(256, 256),
    batch_size=32,
    color_mode="rgb",
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    directory='validation',
    target_size=(256, 256),
    batch_size=32,
    color_mode="rgb",
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Загрузка предобученной модели VIT
vit_model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)

# Заморозка весов модели VIT
vit_model.trainable = False

# Создание модели классификатора
model = Sequential([
    vit_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    Dense(3, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Оценка модели
loss, accuracy = model.evaluate(val_generator)
print('Loss:', loss)
print('Accuracy:', accuracy)

# Сохранение модели
model.save('vit_model.h5')