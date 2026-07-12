import cv2
import pyttsx3
import threading
import time
from ultralytics import YOLO

# Load trained currency model
MODEL_PATH = r"C:\Users\91984\OneDrive\Desktop\SkillArion\runs\Training\classify\train\weights\best.pt"

model = YOLO(MODEL_PATH)

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty("rate", 170)

last_spoken = ""
last_time = 0


def speak(text):
    global last_spoken, last_time

    current_time = time.time()

    # Prevent repeating same note continuously
    if text == last_spoken and current_time - last_time < 3:
        return

    last_spoken = text
    last_time = current_time

    threading.Thread(
        target=_speak_thread,
        args=(text,),
        daemon=True
    ).start()


def _speak_thread(text):
    engine.say(text)
    engine.runAndWait()


# Currency names for voice output
currency_names = {
    "10": "10 rupees note",
    "20": "20 rupees note",
    "50": "50 rupees note",
    "100": "100 rupees note",
    "200": "200 rupees note",
    "500": "500 rupees note",
    "2000": "2000 rupees note",
    "Invalid": "Invalid currency"
}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to open camera")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Classification
    results = model(frame, verbose=False)

    result = results[0]

    class_id = int(result.probs.top1)
    class_name = result.names[class_id]
    confidence = float(result.probs.top1conf)

    if confidence > 0.70:

        display_text = f"{class_name} ({confidence*100:.1f}%)"

        cv2.putText(
            frame,
            display_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        voice_text = currency_names.get(
            class_name,
            f"{class_name} rupees note"
        )

        speak(voice_text)

    cv2.imshow("Currency Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()