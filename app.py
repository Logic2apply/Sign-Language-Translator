from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import base64
import os
import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math


app = Flask(__name__)
CORS(app, support_credentials=True)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20  # used to add a bit of leeway space in accessing the object(hand)
imgSize = 300

# Classifier
classifier = Classifier("converted_keras/keras_model.h5", "converted_keras/labels.txt")

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

# @app.route("/")
# def index():
#     return "<p>Hello, World!</p>"


# @cross_origin(supports_credentials=True)
@app.route("/train", methods=["POST", "GET"])
def trainFromImage():
    if request.method == "POST":
        data = request.get_json()
        image_data = data.get("imageData")
        letter = data.get("letter")

        print(f"letter{letter}")

        if not image_data or not letter:
            return jsonify({"error": "No image data received or no letter given"})

        # Decode base64 image data
        binary_data = base64.b64decode(image_data)

        # open base64 image in opencv
        im_arr = np.frombuffer(
            binary_data, dtype=np.uint8
        )  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        # Draw landmarks
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

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imageWhite)
            filedir = f"images/{letter}"
            if not os.path.exists(filedir):
                os.mkdir(os.path.join(filedir))

            cv2.imwrite(f"{filedir}/{letter}_{time.time()}.jpg", imageWhite)

            # Specify the path where you want to save the image
            # image_path = f"images/image{time.time()}.jpg"

            # Save the image to the specified path
            # with open(image_path, "wb") as f:
            #     f.write(binary_data)

            return jsonify(
                {"message": "Image received and saved successfully", "detection": True}
            )
        return jsonify({"message": "Hand not detected", "detection": False})
    return jsonify({"method": "POST"})


@app.route("/classify", methods=["POST", "GET"])
def classifyimage():
    if request.method == "POST":
        data = request.get_json()
        image_data = data.get("imageData")

        if not image_data:
            return jsonify({"error": "No image data received"})

        # Decode base64 image data
        binary_data = base64.b64decode(image_data)

        # open base64 image in opencv
        im_arr = np.frombuffer(
            binary_data, dtype=np.uint8
        )  # im_arr is one-dim Numpy array
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

        # Draw landmarks
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

            return jsonify(
                {
                    "message": "Image received and saved successfully",
                    "detection": True,
                    "Letter": labels[Index],
                }
            )
        return jsonify({"message": "Hand not detected", "detection": False, "Letter": ""})
    return jsonify({"method": "POST"})


if __name__ == "__main__":
    app.run(debug=True)
