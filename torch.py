import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# PyTorch modelini yükleyin
model = torch.load("model_egitim.pth")  # Yeni model dosyası adı burada
model.eval()  # Modeli değerlendirme moduna alın

# Sınıf etiketleri
labels = {0: 'Devre Kesici', 1: 'Inventer Sürücü', 2: 'Klemens', 3: 'Kontaktör'}

# Görüntü dönüşüm işlemleri
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    height, width, channels = frame.shape

    # YOLO için ön işleme
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Nesne tespitlerini işleme
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Tespit edilen nesneler için NMS uygulayın
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.1)  # NMS parametrelerini düşürdüm

    if len(indexes) > 0:
        indexes = indexes.flatten()  # Düz listeye dönüştürüyoruz

    for i in range(len(boxes)):
        if len(indexes) == 0 or i in indexes:  # indexes listesi boşsa bile kutuları göster
            x, y, w, h = boxes[i]
            
            # ROI'nin geçerli olup olmadığını kontrol edin
            if x < 0 or y < 0 or x + w > width or y + h > height:
                continue  # Geçersiz kutuları atla
            
            roi = frame[y:y+h, x:x+w]
            
            # ROI'nin boş olmadığını kontrol edin
            if roi.size == 0:
                continue  # Geçersiz ROI atla

            # Görüntü dönüştürme işlemi
            roi_transformed = transform(roi)
            roi_transformed = roi_transformed.unsqueeze(0)  # Batch boyutunu ekleyin

            # Parça sınıflandırması yap
            with torch.no_grad():
                predictions = model(roi_transformed)
                predicted_label_idx = predictions.argmax(dim=1).item()
                predicted_label = labels[predicted_label_idx]

            # Çerçeve ve etiket çiz
            label = "{}: {:.2f}%".format(predicted_label, torch.softmax(predictions, dim=1).max().item() * 100)
            color = (0, 255, 0)  # Yeşil renk ile çerçeve çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sonucu göster
    cv2.imshow("Elektrik Kontrol Panosu", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
