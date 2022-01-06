from multiprocessing.queues import Queue
from time import time
import multiprocessing
from typing import Tuple
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf
import cv2
import numpy as np

from voronoi_image_merge import points_to_voronoi
from models import PointDatum, Point

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


def get_closest_point_and_data(
    pointMap: list[PointDatum], p: Point
) -> tuple[PointDatum, float]:
    closest: PointDatum = pointMap[0]
    closestDistSqr: float = 6969

    for point_datum in pointMap:

        distSqr = (point_datum.point.normalized_x - p.normalized_x) ** 2 + (
            point_datum.point.normalized_y - p.normalized_y
        ) ** 2

        if distSqr < closestDistSqr:
            closestDistSqr = distSqr
            closest = point_datum

    return closest, closestDistSqr


def calculatePointMap(
    pointMap: list[PointDatum], points: list[Point], sampleImages: list
) -> list[PointDatum]:

    remainingImages = list.copy(sampleImages)
    pointMap = list.copy(pointMap)
    distanceThreshold = 0.2

    newPointMap: list[PointDatum] = []

    pointsToGiveImages = []
    for point in points:
        if len(remainingImages) == 0:
            break
        if len(pointMap) == 0:
            point_datum = PointDatum()
            point_datum.datum = remainingImages.pop(0)
            point_datum.point = point
            newPointMap.append(point_datum)
            continue

        closest_point, distSqrd = get_closest_point_and_data(pointMap, point)

        if distSqrd < distanceThreshold ** 2:
            new_point = PointDatum()
            new_point.datum = closest_point.datum
            new_point.emotion_label = closest_point.emotion_label
            new_point.emotions_array = closest_point.emotions_array
            new_point.point = point
            newPointMap.append(new_point)
            pointMap.remove(closest_point)

            deleted = False
            for i, im in enumerate(remainingImages):
                if im == closest_point.datum:
                    deleted = True
                    del remainingImages[i]
                    break
            if not deleted:
                print("why you do dis")
        else:
            pointsToGiveImages.append(point)

    for point in pointsToGiveImages:
        if len(remainingImages) == 0:
            print("too many nafarat 2")
            break
        point_datum = PointDatum()
        point_datum.point = point
        point_datum.datum = remainingImages.pop(0)
        newPointMap.append(point_datum)

    return newPointMap


# [[[x,y], image,emotions],[[x,y], image,emotions],]


def doLoop(dataArr, pointMapQueue):

    model = getModel()

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
    pointMap: list[PointDatum] = []
    points: list[Point] = []
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
        points = calculate_points_from_faces(frame, faces)

        if len(points) > 0:
            pointMap = calculatePointMap(pointMap, points, dataArr)
            pointMapQueue.put(pointMap)
            for point_datum in pointMap:
                p = point_datum.point
                roi_gray = gray[p.y : p.y + p.h, p.x : p.x + p.w]
                cropped_img = np.expand_dims(
                    np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0
                )
                prediction = model.predict(cropped_img)
                prediction = prediction[0].tolist()
                point_datum.emotions_array.append(prediction)
                if len(point_datum.emotions_array) >= 5:
                    avg = [0, 0, 0, 0, 0, 0, 0]
                    avg = np.sum(point_datum.emotions_array, 0).tolist()
                    avg = np.divide(avg, 5).tolist()
                    maxindex = int(np.argmax(avg))
                    point_datum.emotion_label = maxindex
                    point_datum.emotions_array.pop(0)
        else:
            pointMapQueue.put([PointDatum()])

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

        cv2.imshow("camera", frame)
        t = time()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    exit()


def calculate_points_from_faces(frame, faces) -> list[Point]:
    points = []
    for (x, y, w, h) in faces:
        midFaceX = x + w / 2
        midFaceY = y + h / 2
        ratioX = frame.shape[1] / (frame.shape[1] - w)
        ratioY = frame.shape[0] / (frame.shape[0] - h)

        point = Point()

        point.normalized_x = (midFaceX / frame.shape[1] * 2 - 1) * ratioX
        point.normalized_y = (midFaceY / frame.shape[0] * 2 - 1) * ratioY

        point.normalized_x = max(min(point.normalized_x, 1), -1)
        point.normalized_y = max(min(point.normalized_y, 1), -1)

        point.x = x
        point.y = y
        point.w = w
        point.h = h

        points.append(point)
    return points


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

    pointMapQueue: Queue = multiprocessing.Queue()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    camera_thread = multiprocessing.Process(
        target=doLoop, args=([0, 1, 2], pointMapQueue), daemon=True
    )
    camera_thread.start()

    targetPM: list[PointDatum] = []
    while True:
        cv2.waitKey(1)
        while pointMapQueue.qsize() > 1:
            targetPM = pointMapQueue.get()

        if len(targetPM) > 0:
            finalImage = points_to_voronoi(
                [sampleImages[point_datum.datum] for point_datum in targetPM],
                np.array(
                    [
                        [point_datum.point.normalized_x, point_datum.point.normalized_y]
                        for point_datum in targetPM
                    ]
                ),
                renderDots=True,
            )
            cv2.imshow(
                "vornoi",
                finalImage,
            )
