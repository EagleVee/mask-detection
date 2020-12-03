#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
import time
import face_recognition

filepath = 'model/mask1_model_resnet101.pth'
model = torch.load(filepath, map_location=torch.device('cpu'))
model.eval()
model.cpu()
device = torch.device("cpu")
cascPath = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

class_names = ['with_mask', 'without_mask']


def process_image(image):
    # pil_image = Image.open(image)
    pil_image = image

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = image_transforms(pil_image)
    return img


def classify_face(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(image)
    image = process_image(im)
    img = image.unsqueeze_(0)
    img = image.float()
    output = model(image)
    _, predicted = torch.max(output, 1)

    classification1 = predicted.data[0]
    index = int(classification1)
    print('PREDICTED ', class_names[index])
    return class_names[index]


def detect_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lấy ra vị trí khuôn mặt
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(15, 15)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = image[y: y + h, x: x + w]

        # Cho model detect, trả về "with_mask" hoặc "without_mask"
        label = classify_face(cropped_face)
        cv2.putText(image, str(label), (int(x), int(y - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    image = cv2.imread('crowd_image.jpg')
    detected_image = detect_mask(image)
    cv2.imshow('Result', detected_image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    # KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')
    #
    # image = face_recognition.load_image_file("crowd_image_2.jpg")
    # faces = face_recognition.face_locations(image)
    #
    # for (top, right, bottom, left) in faces:
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    #     cropped_face = image[top: bottom, left:right]
    #     label = classify_face(cropped_face)
    #     cv2.putText(image, str(label), (int(left), int(top - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
