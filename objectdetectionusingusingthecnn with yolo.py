# Step 1: Install required packages (run once)
!pip install ultralytics pyttsx3 opencv-python-headless
# -----------------------------
# Step 2: Import libraries
# -----------------------------
import cv2
import pyttsx3
import threading, queue
import glob, os
from ultralytics import YOLO

# -----------------------------
# Step 3: Load YOLO model
# -----------------------------
# Change yolov8n.pt to yolov8s.pt for better accuracy
model = YOLO("yolov8n.pt")

# -----------------------------
# Step 4: Set dataset path
# -----------------------------
dataset_path = r"C:\Users\Laxmi Priya\Downloads\archive (1)\coco2017\val2017"  # change to your path
print("Path exists?", os.path.exists(dataset_path))

# -----------------------------
# Step 5: Setup TTS safely
# -----------------------------
engine = pyttsx3.init()
speech_queue = queue.Queue()

def tts_loop():
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()

# Run TTS in a background thread
threading.Thread(target=tts_loop, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# -----------------------------
# Step 6: Get image files
# -----------------------------
images = glob.glob(os.path.join(dataset_path, "*.jpg"))
print("Total images found:", len(images))

# ✅ Limit number of images to avoid overload (optional)
images = images[:20]
print("Processing images:", len(images))

# -----------------------------
# Step 7: Object detection loop
# -----------------------------
for img_path in images:
    # Run YOLO detection
    results = model(img_path)

    # Load image for visualization
    img = cv2.imread(img_path)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)

            # Speak object name
            speak(f"Detected {label}")

    # Show image
    cv2.imshow("Detection", img)
    if cv2.waitKey(500) & 0xFF == ord('q'):  # press 'q' to quit early
        break

cv2.destroyAllWindows()
print("Detection finished!")