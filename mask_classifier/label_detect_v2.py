import numpy as np
import cv2
import random

filepath = 'model/mask1_model_resnet101.pth'
face_casc_path = 'haarcascade_frontalface_default.xml'
eye_casc_path = 'haarcascade_eye.xml'
mouth_casc_path = 'haarcascade_mcs_mouth.xml'
upper_body_casc_path = 'haarcascade_upperbody.xml'

faceCascade = cv2.CascadeClassifier(face_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)
mouthCascade = cv2.CascadeClassifier(mouth_casc_path)
upperBodyCascade = cv2.CascadeClassifier(upper_body_casc_path)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def classify_face(image):
    mouths = mouthCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(mouths) == 0:
        return 'with_mask'
    else:
        return 'without_mask'


def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = gray[y: y + h, x: x + w]

        # Trả về "with_mask" hoặc "without_mask"
        label = classify_face(cropped_face)
        cv2.putText(image, str(label), (int(x), int(y - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    image = cv2.imread('crowd_image_2.jpg')
    detected_image = detect_mask(image)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
