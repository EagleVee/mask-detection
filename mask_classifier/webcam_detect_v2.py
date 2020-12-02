import cv2
from label_detect_v2 import detect_mask

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    detected_frame = detect_mask(frame)
    cv2.imshow('Result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
