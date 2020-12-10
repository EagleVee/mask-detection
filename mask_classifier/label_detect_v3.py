import numpy as np
import cv2

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([255, 173, 127], np.uint8)

face_casc_path = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(face_casc_path)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def has_mask(image, w, h):
    face_area = w * h
    face_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    skin_region = cv2.inRange(face_ycrcb, min_YCrCb, max_YCrCb)
    contours, hierarchy = cv2.findContours(skin_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        cv2.drawContours(image, contours, i, (0, 255, 0), 1)
        if area > face_area * 0.5:
            return False

    return True


def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
    )

    for (x, y, w, h) in faces:
        cropped_face = image[y: y + h, x: x + w]
        face_area = w * h
        face_ycrcb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2YCR_CB)
        skin_region = cv2.inRange(face_ycrcb, min_YCrCb, max_YCrCb)
        contours, hierarchy = cv2.findContours(skin_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        label = 'not_detected'
        has_skin = False
        for i, c in enumerate(contours):
            has_skin = True
            area = cv2.contourArea(c)
            cv2.drawContours(cropped_face, contours, i, (0, 255, 0), 1)
            if area < face_area * 0.5:
                label = 'with_mask'
            else:
                label = 'without_mask'

        if has_skin:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, str(label), (int(x), int(y - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    image = cv2.imread('crowd_image.jpg')
    detected_image = detect_mask(image)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
