import cv2
from label_detect import detect_mask

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    detected_frame = detect_mask(frame)

    cv2.imshow('Mask Detector', detected_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
