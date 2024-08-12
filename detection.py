import cv2
import numpy as np

# MobileNet SSD model dosyaları
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

# Sınıf etiketleri
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "controlpanel"]

# Renkler
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Modeli yükle
try:
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
except cv2.error as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    exit()

# Kamerayı başlat
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

    # Giriş boyutlarını ayarla
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Blob'u ağ üzerinden geç
    net.setInput(blob)
    try:
        detections = net.forward()
    except cv2.error as e:
        print(f"İleri geçiş sırasında hata oluştu: {e}")
        break

    # Tespit edilen objeleri işle
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Güven eşiğini kontrol et
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Çerçeve ve etiket çiz
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Sonucu göster
    cv2.imshow("Frame", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()