!pip install ultralytics
!pip install gTTS
import os
os.getcwd()
os.listdir()
!pip install -q ultralytics pyttsx3 opencv-python
import cv2
import glob
import pyttsx3
import threading
from ultralytics import YOLO
import os
# Path to your COCO2017 images (change this)
dataset_path = r"C:\Users\Laxmi Priya\Downloads\archive (1)\coco2017\val2017"  # Windows example
# dataset_path = "/home/user/Downloads/coco2017/val2017"        # Linux/Mac example

# Load pretrained YOLO model (CNN-based)
model = YOLO("yolov8n.pt")   # lightweight model; try yolov8s.pt for more accuracy
import os, glob

print("Using path:", dataset_path)
print("Path exists?", os.path.exists(dataset_path))

images = glob.glob(os.path.join(dataset_path, "*.jpg"))
print("Found", len(images), "images")
print("First 5 images:", images[:5])
def speak_text(text):
    """Speak text asynchronously so detection loop doesn’t freeze."""
    def _speak(t):
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(t)
        engine.runAndWait()
        engine.stop()
    threading.Thread(target=_speak, args=(f"Detected {label}",)).start()
import cv2
import pyttsx3
import threading, queue
import glob, os

# ----------------------------
# Setup TTS (only one thread)
# ----------------------------
engine = pyttsx3.init()
speech_queue = queue.Queue()

def tts_loop():
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()

# start one background thread
threading.Thread(target=tts_loop, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ----------------------------
# Step 5: Run detection
# ----------------------------
images = glob.glob(os.path.join(dataset_path, "*.jpg"))
print("Total images in dataset:", len(images))

# Limit to first 20 images (you can change this number)
images = images[:20]
print("Now processing only:", len(images), "images")

for img_path in images:
    # Run YOLO detection
    results = model(img_path)

    # Load the image
    img = cv2.imread(img_path)

    # Loop over detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # draw box + label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)

            # speak object name
            speak(f"Detected {label}")

    # Show image
    cv2.imshow("Detection", img)

    # wait for key press (press q to quit early)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
