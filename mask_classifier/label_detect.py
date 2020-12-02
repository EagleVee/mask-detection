#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import datasets, models, transforms
from PIL import Image
import cv2
import time

filepath = 'model/mask1_model_resnet101.pth'
model = torch.load(filepath, map_location=torch.device('cpu'))
model.eval()
model.cpu()
device = torch.device("cpu")
cascPath = 'haarcascade_frontalface_default.xml'
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
    print('START', time.time())
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
    print('END', time.time())
    return class_names[index]


if __name__ == '__main__':
    image = cv2.imread('crowd_image_2.jpg')
    print('processed image', image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Lấy ra vị trí khuôn mặt
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = image[y: y + h, x: x + w]

        # Cho model detect, trả về "with_mask" hoặc "without_mask"
        label = classify_face(cropped_face)
        cv2.putText(image, str(label), (int(x), int(y - 10)), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Result', image)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
