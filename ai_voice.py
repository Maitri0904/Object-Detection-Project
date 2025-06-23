#######    object detection and ai voice generate is done 

from ultralytics import YOLO
import cv2
import pyttsx3
import time

# Load YOLOv8 model (change this to your custom model path if needed)
model = YOLO("yolov8n.pt")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Connect to phone camera
cap = cv2.VideoCapture('http://192.168.29.152:8080/video')

# For managing repeated announcements
last_spoken = ""
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame, imgsz=640)[0]
    annotated_frame = results.plot()

    # Loop through detected objects
    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]

        # Speak only if it's a new object or enough time has passed
        if label != last_spoken or time.time() - last_time > 3:
            print(f"Detected: {label}")
            engine.say(f"{label} ahead")
            engine.runAndWait()
            last_spoken = label
            last_time = time.time()

    # Display the frame with detections
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()