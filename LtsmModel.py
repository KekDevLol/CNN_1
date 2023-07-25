import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models

# Создание генератора изображений
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory='train',
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",  #
    class_mode='categorical',
    subset='training',
    seed=42
)

val_generator = train_datagen.flow_from_directory(
    directory='validation',
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical',
    subset='validation',
    seed=42
)

# Определение архитектуры
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.TimeDistributed(layers.Flatten()))# Преобразует выходы из сверточных слоев в 3-х мерный формат,необходимый для LTSM
model.add(layers.LSTM(256))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

# Компиляция
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Обучение
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20
)

# Сохранение
model.save('LTSM_model.h5')