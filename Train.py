from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "../dataset/FER2013/train",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

validation_generator = train_data.flow_from_directory(
    "../dataset/FER2013/test",
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

model = create_model()

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30
)

model.save("../models/emotion_model.h5")
