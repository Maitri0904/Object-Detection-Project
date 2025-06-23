#  python with phone connection and object detection ##########################################


from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # or your trained model path
cap = cv2.VideoCapture('http://192.168.91.166:8080/video')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=640)[0]
    annotated_frame = results.plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()































