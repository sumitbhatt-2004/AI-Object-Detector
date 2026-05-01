import cv2
from ultralytics import YOLO

# 1. Load the pre-trained model (Nano version is fast for laptops)
model = YOLO('yolov8n.pt') 

# 2. Open the Laptop Webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit the camera window")

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    if success:
        # 3. Run YOLO detection on the current frame
        # conf=0.5 matlab 50% sure hone par hi box dikhayega
        results = model(frame, conf=0.5)

        # 4. Get the annotated frame (boxes and labels)
        annotated_frame = results[0].plot()

        # 5. Display the frame
        cv2.imshow("Deep Learning Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
