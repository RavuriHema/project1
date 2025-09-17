!pip install pyttsx3
# Step 1: Install required packages (run once)
!pip install ultralytics pyttsx3 opencv-python-headless

# -----------------------------
# Step 2: Import libraries
# -----------------------------
import cv2
import pyttsx3
import threading, queue
from ultralytics import YOLO

# -----------------------------
# Step 3: Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")  # use yolov8s.pt for better accuracy

# -----------------------------
# Step 4: Setup TTS safely
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
# Step 5: Video path
# -----------------------------
video_path = r"C:\Users\Laxmi Priya\OneDrive\Desktop\VITW\dog.mp4"  # change to your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    print("Video opened successfully!")

# -----------------------------
# Step 6: Object detection loop
# -----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            # Speak object name
            speak(f"Detected {label}")

    # Display frame
    cv2.imshow("Video Detection", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Video detection finished!")