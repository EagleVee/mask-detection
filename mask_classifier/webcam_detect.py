import cv2
from label_detect import classify_face

cascPath = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = frame[y: y+h, x: x + w]
        # Cho model detect, trả về "with_mask" hoặc "without_mask"
        label = classify_face(cropped_face)
        cv2.putText(frame, str(label), (int(x), int(y - 10)), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Mask Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
