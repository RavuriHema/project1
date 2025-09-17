# -----------------------------
# Step 1: Install required packages
# -----------------------------
!pip install opencv-python pyttsx3 --quiet
import os
os.getcwd()
os.listdir('Downloads/archive (1)')
os.chdir('Downloads/archive (1)/coco2017')
os.listdir()
import cv2
import numpy as np
import pyttsx3
import os
# -----------------------------
# Step 3: Initialize Text-to-Speech
# -----------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # speed of speech
# -----------------------------
# Step 4: Load MobileNet SSD model
# -----------------------------
prototxt_path = r"MobileNetSSD_deploy.prototxt"
model_path = r"MobileNetSSD_deploy.caffemodel"

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
# -----------------------------
# Step 5: Class labels MobileNet SSD can detect
# -----------------------------
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# -----------------------------
# Step 6: Dataset folder
# -----------------------------
dataset_folder = "val2017"  # your dataset folder
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)
for i in os.listdir(dataset_folder):
    print(i)
# -----------------------------
# Step 7: Process each image in dataset
# -----------------------------
for i
mg_name in os.listdir(dataset_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(dataset_folder, img_name)
        image = cv2.imread(img_path)
        orig_image = image.copy()
        (h, w) = image.shape[:2]

        # Prepare image for MobileNet SSD
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # confidence threshold
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(orig_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(orig_image, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Speak object name
                engine.say(CLASSES[idx])
                engine.runAndWait()

        # Save output image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, orig_image)

