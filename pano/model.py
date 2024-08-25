from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Model tanımı
model = Sequential([
    MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    Flatten(),
    Dense(1024, activation='relu'),
    Dense(4, activation='softmax')  # Kategori sayısına göre ayarlayın
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Veri hazırlama
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'pano/data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'pano/data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Modeli eğitme
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator
)

# Modeli kaydetme
model.save('pano_modeli3.keras')
