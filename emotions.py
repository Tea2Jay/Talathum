import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
from time import sleep
from tensorflow.python.keras.backend import dtype

from tensorflow.python.keras.utils.generic_utils import to_list
from realTimeLatentWalk import LatentWalkerController
from voronoi_image_merge import points_to_voronoi

# controller = LatentWalkerController()


# Hide GPU from visible devices
tf.config.set_visible_devices([], "GPU")


def getModel():
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1))
    )
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))

    # emotions will be displayed on your face from the webcam feed
    model.load_weights("model.h5")
    return model


# Create the model
model = getModel()
# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
# emotion_dict = {0: "Angry", 1: "Happy", 2: "Sad"}
emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}


# start the webcam feed
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap = cv2.VideoCapture("C:/Users/3izzo/Desktop/Projects/Talathum/WIN_20211021_23_20_21_Pro.mp4")
facecasc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# sleep(20)
# sleep(1)
sampleImages = [
    cv2.resize(cv2.imread(path), (512, 512))
    for path in [
        "images/1024/3970-00.png",
        "images/1024/822737-00.png",
        "images/Emotions/Anger/Seated Nude.jpg",
        "images/Emotions/Anger/The Family.jpg",
    ]
]

prevPoints = []
points = []
emotions = []
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # print(faces)
    prevPoints = points
    points = []

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        cropped_img = np.expand_dims(
            np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
        )
        prediction = model.predict(cropped_img)
        prediction = prediction[0].tolist()
        emotions.append(prediction)
        if len(emotions) >= 5:
            avg = [0, 0, 0, 0, 0, 0, 0]
            avg = np.sum(emotions, 0).tolist()
            avg = np.divide(avg, 5).tolist()
            maxindex = int(np.argmax(avg))
            cv2.putText(
                frame,
                emotion_dict[maxindex],
                (x + 20, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            emotions.pop(0)

        midFaceX = x + w / 2
        midFaceY = y + h / 2
        ratioX = frame.shape[1] / (frame.shape[1] - w)
        ratioY = frame.shape[0] / (frame.shape[0] - h)
        point = [
            (midFaceX / frame.shape[1] * 2 - 1) * ratioX,
            (midFaceY / frame.shape[0] * 2 - 1) * ratioY,
        ]

        for i in range(2):
            if point[i] > 1:
                point[i] = 1
            elif point[i] < -1:
                point[i] = -1

        points.append(point)
        # prediction = prediction[0]
        # prediction = [prediction[0], prediction[3], prediction[5]]
        # emotions = [emotions[0]+prediction[0], emotions[1] +
        #             prediction[1], emotions[2]+prediction[2]]

    cv2.imshow("Video", cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC))
    if len(points) > 0:
        cv2.imshow(
            "vornoi", points_to_voronoi(sampleImages, np.array(points), renderDots=True)
        )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
exit()
