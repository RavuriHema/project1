pip install opencv-python
pip install pyttsx3
!pip install torch torchvision opencv-python pyttsx3
# Run this cell to start webcam detection with voice
import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO

# ----------- User parameters -------------
MODEL_NAME = "yolov8n.pt"   # small model; change to yolov8s.pt / yolov8m.pt if you want higher accuracy (needs more GPU/CPU)
CONFIDENCE_THRESHOLD = 0.35
SPEAK_COOLDOWN_SECONDS = 2.0   # minimum seconds before speaking same label again
WEBCAM_INDEX = 0               # change to 1,2... if your webcam is on a different index
IMG_SIZE = 640                 # inference image size (ultralytics will auto-scale)
# ----------------------------------------

# Load model (downloads automatically if not present)
print("Loading model. This may take a moment...")
model = YOLO(MODEL_NAME)

# Track when labels were last spoken to avoid repeating too often
last_spoken = {}  # {label: timestamp}

def speak_text(text):
    """
    Plays the text-to-speech in a separate thread.
    Uses a fresh pyttsx3 engine instance per thread to avoid blocking UI.
    """
    def _speak(t):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # speech rate
            engine.say(t)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            # TTS failed; just ignore to avoid crashing the main loop
            print("TTS error:", e)

    th = threading.Thread(target=_speak, args=(text,), daemon=True)
    th.start()

def process_frame(frame):
    """
    Run model on a single frame and return annotated frame and list of detections.
    Each detection: (label, confidence, (x1,y1,x2,y2))
    """
    # YOLO accepts BGR frames directly in ultralytics implementation
    results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False, device='cpu') 
    # results is a list; we used single frame so results[0]
    detections = []
    annotated = frame.copy()

    if len(results) > 0:
        r = results[0]
        # r.boxes has xyxy, score, cls
        boxes = getattr(r, 'boxes', None)
        if boxes is not None:
            for box in boxes:
                # box.xyxy -> tensor/numpy with 4 coords
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0].numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) else float(box.conf)
                cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) else int(box.cls)
                label = model.model.names[cls_id] if hasattr(model, 'model') and hasattr(model.model, 'names') else str(cls_id)
                
                # draw box and label
                text = f"{label} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # put text background for readability
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), (0,255,0), -1)
                cv2.putText(annotated, text, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

                detections.append((label, conf, (x1, y1, x2, y2)))
    return annotated, detections

def main_loop():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
        return

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            annotated, detections = process_frame(frame)

            # speak detected labels with cooldown
            now = time.time()
            spoken_this_frame = set()
            for label, conf, bbox in detections:
                # only speak labels above threshold and not repeated too fast
                last = last_spoken.get(label, 0)
                if (now - last) >= SPEAK_COOLDOWN_SECONDS and label not in spoken_this_frame:
                    # speak asynchronously
                    speak_text(label)
                    last_spoken[label] = now
                    spoken_this_frame.add(label)

            # show frame
            cv2.imshow("YOLO Webcam Detection (press q to quit)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Exiting...")

# Run the loop
main_loop()