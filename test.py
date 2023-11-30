import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras", "model_label")


offset = 20  # used to add a bit of leeway space in accessing the object(hand)
imgSize = 300
labels = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]

        imageWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 225

        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]

        imgCropShape = imgCrop.shape

        aspectratio = h / w
        if aspectratio > 1:
            k = imgSize / h
            w_calc = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (w_calc, imgSize))
            imgResizeShape = imgResize.shape
            W_gap = math.ceil((imgSize - w_calc) / 2)
            imageWhite[:, W_gap : w_calc + W_gap] = imgResize
            Prediction, Index = classifier.getPrediction(imageWhite, draw=False)
            print(Prediction, Index)

        if aspectratio < 1:
            k = imgSize / w
            h_calc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, h_calc))
            imgResizeShape = imgResize.shape
            H_gap = math.ceil((imgSize - h_calc) / 2)
            imageWhite[H_gap : h_calc + H_gap, :] = imgResize
            Prediction, Index = classifier.getPrediction(imageWhite, draw=False)
            print(Prediction, Index)

        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x - offset + 90, y - offset),
            (57, 255, 20),
            cv2.filled,
        )
        cv2.putText(
            imgOutput,
            labels[Index],
            (x, y - 26),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (57, 255, 20),
            2,
        )
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (57, 255, 20),
            4,
        )

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
