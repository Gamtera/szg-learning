import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Veri artırma (data augmentation) işlemi ile eğitim verilerini hazırlama
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # %20 validation set
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

# Eğitim verilerini yükleme
train_generator = datagen.flow_from_directory(
    'dataset_directory',  # Görüntülerin olduğu klasör
    target_size=(224, 224),  # Görüntülerin yeniden boyutlandırılması
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Doğrulama verilerini yükleme
validation_generator = datagen.flow_from_directory(
    'dataset_directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
