import numpy as np
import cv2
import random

face_casc_path = 'cascades/haarcascade_frontalface_default.xml'
face_alt_casc_path = 'cascades/haarcascade_frontalface_alt.xml'
face_alt_2_casc_path = 'cascades/haarcascade_frontalface_alt2.xml'
eye_casc_path = 'cascades/haarcascade_eye.xml'
mouth_casc_path = 'cascades/haarcascade_mcs_mouth.xml'
nose_casc_path = 'cascades/nose.xml'

faceCascade = cv2.CascadeClassifier(face_casc_path)
faceAltCascade = cv2.CascadeClassifier(face_alt_casc_path)
faceAlt2Cascade = cv2.CascadeClassifier(face_alt_2_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)
mouthCascade = cv2.CascadeClassifier(mouth_casc_path)
noseCascade = cv2.CascadeClassifier(nose_casc_path)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = gray[y: y + h, x: x + w]

        mouths = mouthCascade.detectMultiScale(
            cropped_face,
            scaleFactor=2,
            minNeighbors=5,
            minSize=(int(w / 3), int(h / 15))
        )

        # eyes = eyeCascade.detectMultiScale(
        #     cropped_face,
        #     scaleFactor=2,
        #     minNeighbors=5
        # )
        #
        #
        # for (nx, ny, nw, nh) in eyes:
        #     if ny > h / 2:
        #         is_face_reverted = True
        #         cv2.rectangle(image, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (0, 0, 255), 1)

        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(image, (x + mx, y + my), (x + mx + mw, y + my + mh), (255, 0, 0), 1)

        if len(mouths) == 0:
            label = 'with_mask'
        else:
            label = 'without_mask'

        print('PREDICTED', label)

        cv2.putText(image, str(label), (int(x), int(y - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    image = cv2.imread('test_image_without_mask.jpg')
    detected_image = detect_mask(image)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
