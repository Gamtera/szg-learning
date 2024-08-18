import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Modeli yükle (SavedModel formatında ya da h5 formatında olabilir)
model_path = ("my_model.keras")
model = tf.keras.models.load_model(model_path)

# Etiketleri yükle
class_names = open("labels.txt", "r").readlines()

# Web kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü yeniden boyutlandır ve normalleştir
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    size = (224, 224)  # Modelinize uygun boyut
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Görüntüyü numpy array'e çevir ve normalleştir
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Modelin beklediği formatta array oluştur
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Model ile tahmin yap
    predictions = model.predict(data)
    index = np.argmax(predictions)
    class_name = class_names[index].strip()
    confidence_score = predictions[0][index]

    # Sonucu ekrana yazdır
    label = f"{class_name} ({confidence_score:.2f})"
    
    # Görüntü üzerinde sonucu göster
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Görüntüyü göster
    cv2.imshow('Panel Detection', frame)
    
    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
