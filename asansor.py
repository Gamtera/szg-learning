import cv2
import numpy as np
import tensorflow as tf

# TensorFlow model dosyasını yükleyin (modelinizin yolu ile değiştirin)
model_path = "keras_model.h5"
model = tf.keras.models.load_model(model_path)

# Etiketleri dosyadan yükleyin
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return labels

labels_file = "labels.txt"
LABELS = load_labels(labels_file)

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Giriş boyutlarını ayarla
    (h, w) = frame.shape[:2]

    # Gerekli görüntü ön işleme adımları (örneğin, yeniden boyutlandırma ve normalize etme)
    processed_frame = cv2.resize(frame, (224, 224))  # Modelin kabul ettiği giriş boyutlarına göre ayarlayın
    processed_frame = processed_frame / 255.0  # Normalizasyon
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Modelin giriş şekline uygun hale getirin

    # Model üzerinden tahmin yapın
    predictions = model.predict(processed_frame)
    confidence = np.max(predictions)
    label_index = np.argmax(predictions)

    if confidence > 0.5:  # Güven eşiği
        label = f"{LABELS[label_index]}: {confidence * 100:.2f}%"
        color = (0, 255, 0) if label_index == 0 else (0, 0, 255)  # Asansör paneli için yeşil, diğerleri için kırmızı

        # Çerçeve ve etiket çiz
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, 2)
        y = 30
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Sonucu göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
#deneme