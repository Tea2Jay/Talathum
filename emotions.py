import os
from threading import Thread
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
from time import sleep, time, process_time_ns
from tensorflow.python.keras.backend import dtype

from tensorflow.python.keras.utils.generic_utils import to_list
from realTimeLatentWalk import LatentWalkerController
from voronoi_image_merge import points_to_voronoi

# controller = LatentWalkerController()


def getModel():

    # Hide GPU from visible devices
    tf.config.set_visible_devices([], "GPU")

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


# sleep(20)
# sleep(1)


def get_closest_point_and_data(pointMap, p):
    closest = pointMap[0]
    closestDistSqr = 6969

    for [point, image, emotions] in pointMap:
        distSqr = (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2
        if distSqr < closestDistSqr:
            closestDistSqr = distSqr
            closest = [point, image, emotions]

    return closest[0], closest[1], closest[2], closestDistSqr


def calculatePointMap(pointMap, points, sampleImages):
    remainingImages = list.copy(sampleImages)
    pointMap = list.copy(pointMap)
    distanceThreshold = 0.2

    newPointMap = []

    pointsToGiveImages = []
    for p in points:
        if len(remainingImages) == 0:
            print("too many nafarat")
            break
        if len(pointMap) == 0:
            newPointMap.append([p, remainingImages.pop(0), [[], 0]])
            continue

        closestPoint, image, emotions, distSqrd = get_closest_point_and_data(
            pointMap, p
        )

        if distSqrd < distanceThreshold ** 2:
            newPointMap.append([p, image, emotions])
            pointMap.remove([closestPoint, image, emotions])

            deleted = False
            for i, im in enumerate(remainingImages):
                if im == image:
                    deleted = True
                    del remainingImages[i]
                    break
            if not deleted:
                print("why you do dis")
        else:
            print(f"{p=} {closestPoint=} {distSqrd=}")
            pointsToGiveImages.append(p)

    for p in pointsToGiveImages:
        if len(remainingImages) == 0:
            print("too many nafarat 2")
            break
        newPointMap.append([p, remainingImages.pop(0), [[], 0]])

    return newPointMap


# [[[x,y], image,emotions],[[x,y], image,emotions],]


def doLoop(dataArr, pointMapQueue):
    # global pointMap
    # pointMap = [
    #     [[-0.5, -0.5], dataArr[0], []],
    #     [[0.5, 0.5], dataArr[1], []],
    #     [[-0.25, 0.25], dataArr[2], []],
    # ]

    # while True:
    #     sleep(0.1)
    #     for i in range(3):
    #         pointMap[i][0][0] += np.random.random() * 0.2 - 0.1
    #         pointMap[i][0][1] += np.random.random() * 0.2 - 0.1
    # Create the model
    model = getModel()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

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
    pointMap = []
    points = []
    t = time()
    print("starting camera loop")
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # print(faces)
        points = []
        for (x, y, w, h) in faces:

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

            point.append(x)
            point.append(y)
            point.append(w)
            point.append(h)
            points.append(point)

        # cv2.imshow("Video", cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC))
        if len(points) > 0:
            pointMap = calculatePointMap(pointMap, points, dataArr)
            pointMapQueue.put(pointMap)
            for point, image, emotions in pointMap:
                [x, y, w, h] = point[2:]
                # cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                cropped_img = np.expand_dims(
                    np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
                )
                prediction = model.predict(cropped_img)
                prediction = prediction[0].tolist()
                emotions[0].append(prediction)
                if len(emotions[0]) >= 5:
                    avg = [0, 0, 0, 0, 0, 0, 0]
                    avg = np.sum(emotions[0], 0).tolist()
                    avg = np.divide(avg, 5).tolist()
                    maxindex = int(np.argmax(avg))
                    emotions[1] = maxindex
                    # cv2.putText(
                    #     frame,
                    #     emotion_dict[maxindex],
                    #     (x + 20, y - 60),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (255, 255, 255),
                    #     2,
                    #     cv2.LINE_AA,
                    # )
                    emotions[0].pop(0)

        dt = time() - t
        if dt == 0:
            dt = 0.0001
        # print(f"camera FPS {1/dt}")
        cv2.putText(
            frame,
            str(int(1 / dt)),
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # cv2.imshow("camera", frame)
        t = time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    exit()


if __name__ == "__main__":
    sampleImages = [
        cv2.resize(cv2.imread(path), (512, 512))
        for path in [
            "images/1024/3970-00.png",
            "images/1024/822737-00.png",
            "images/Emotions/Anger/Seated Nude.jpg",
            "images/Emotions/Anger/The Family.jpg",
        ]
    ]

    camera_thread = Thread(target=doLoop, args=([sampleImages]))
    camera_thread.start()
    t = time()
    while True:
        print(1 / (time() - t + 0.000001))
        t = time()
        cv2.waitKey(1)
        if len(pointMap) > 0:
            finalImage = points_to_voronoi(
                [im for _, im, _ in pointMap],
                np.array([point[0:2] for point, _, _ in pointMap]),
                renderDots=True,
            )
            cv2.imshow(
                "vornoi",
                finalImage,
            )
