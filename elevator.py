import cv2
import numpy as np
import tensorflow as tf

# TensorFlow modelinizi yükleyin
model = tf.keras.models.load_model("my_model.keras")  # Model dosyanızın adını buraya yazın
labels = ["elevator", "not elevator"]

# Kamerayı başlatma fonksiyonları (önceki kodda olduğu gibi)
def list_available_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def initialize_camera():
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("Hiçbir kamera açılamadı.")
        return None

    print("Kullanılabilir kameralar: ", available_cameras)
    camera_index = int(input("Kullanmak istediğiniz kamera indeksini girin: "))
    if camera_index not in available_cameras:
        print(f"Geçersiz kamera indeksi: {camera_index}")
        return None

    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Kamera {camera_index} ile açıldı.")
            return cap
        else:
            print(f"Kamera {camera_index} ile açıldı ama görüntü alınamadı.")
    else:
        print(f"Kamera {camera_index} ile açılamadı.")
    return None

cap = initialize_camera()
if cap is None:
    exit()

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    # Giriş boyutlarını ayarla ve ön işlem
    input_image = cv2.resize(frame, (224, 224))  # Modelinizin kabul ettiği boyutlara göre ayarlayın
    input_image = np.expand_dims(input_image, axis=0) / 255.0  # Normalize edin

    # Tahmin yap
    predictions = model.predict(input_image)
    predicted_label = labels[np.argmax(predictions)]

    # Çerçeve ve etiket çiz
    label = "{}: {:.2f}%".format(predicted_label, np.max(predictions) * 100)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
