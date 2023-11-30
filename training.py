import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20  # used to add a bit of leeway space in accessing the object(hand)
imgSize = 300

while True:
    success, img = cap.read()
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
        if aspectratio < 1:
            k = imgSize / w
            h_calc = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, h_calc))
            imgResizeShape = imgResize.shape
            H_gap = math.ceil((imgSize - h_calc) / 2)
            imageWhite[H_gap : h_calc + H_gap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
