import cv2

# For USB webcam (index 0 = first camera)
cap = cv2.VideoCapture(0)

# For IP camera (replace with your IP stream URL)
# cap = cv2.VideoCapture("rtsp://username:password@ip:554/stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()