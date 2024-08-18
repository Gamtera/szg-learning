import cv2
import numpy as np
import tensorflow as tf

# YOLO modelini yükleyin
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]



# TensorFlow modelinizi yükleyin
model = tf.keras.models.load_model("my_model.keras")
labels = ["elevator", "not elevator"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break

    height, width, channels = frame.shape

    # YOLO için ön işleme
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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0
            roi_resized = np.expand_dims(roi_resized, axis=0)

            # Sınıflandırma yap
            predictions = model.predict(roi_resized)
            predicted_label = labels[np.argmax(predictions)]

            # Çerçeve ve etiket çiz
            label = "{}: {:.2f}%".format(predicted_label, np.max(predictions) * 100)
            color = (0, 255, 0) if predicted_label == "elevator" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sonucu göster
    cv2.imshow("Image", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
