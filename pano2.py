import cv2
import numpy as np
import tensorflow as tf

# Model ve etiketleri yükleme
model = tf.keras.models.load_model('my_model.keras')
category_index = {0: 'Devre Kesici', 1: 'Inventor', 2: 'Klemens', 3: 'Kontaktor'}

# Pencere boyutu ve kaydırma adımı
window_size = 128
step_size = 32

# Kamera açma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntü kopyası
    original_frame = frame.copy()
    
    # Kayan pencere ile görüntüde gezme
    for y in range(0, frame.shape[0] - window_size, step_size):
        for x in range(0, frame.shape[1] - window_size, step_size):
            # Pencereyi kırpma
            window = frame[y:y + window_size, x:x + window_size]

            # Pencereyi 224x224 boyutuna yeniden boyutlandırma
            resized_window = cv2.resize(window, (224, 224))

            # Görüntüyü TensorFlow modeline gönderme
            input_tensor = tf.convert_to_tensor([resized_window], dtype=tf.float32)
            input_tensor = input_tensor / 255.0

            # Tahminleri alma
            predictions = model(input_tensor)
            predicted_class = np.argmax(predictions, axis=1)[0]
            label = category_index[predicted_class]

            # Eğer belirli bir sınıf tespit edilirse dikdörtgen çizme
            if predicted_class in category_index:
                cv2.rectangle(original_frame, (x, y), (x + window_size, y + window_size), (0, 255, 0), 2)
                cv2.putText(original_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ekranda görüntüleme
    cv2.imshow('Detection', original_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
