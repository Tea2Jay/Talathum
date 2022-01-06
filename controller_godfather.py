import multiprocessing
from multiprocessing.queues import Queue
from threading import Thread
from time import sleep, time, time_ns
from typing import List
import cv2
from realTimeLatentWalk import LatentWalkerController
import emotions
from voronoi_image_merge import points_to_voronoi
import numpy as np
from models import PointDatum, Point


def smooth(p1: Point, p2: Point, factor) -> Point:
    delta = p2 - p1
    deltaFactored = delta * factor
    res = p1 + deltaFactored
    return res


def smoothPoints(target: list[PointDatum], current: list[PointDatum]):
    lerpFactor = 0.9

    newPM: list[PointDatum] = list.copy(target)

    for i, point_datum in enumerate(newPM):
        for point_datum2 in current:
            if point_datum.datum == point_datum2.datum:
                newPM[i].point = smooth(
                    point_datum.point, point_datum2.point, lerpFactor
                )
                break
    return newPM


if __name__ == "__main__":

    # cv2.namedWindow("vornoi", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("vornoi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    emotionsMap = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2, 5: 2, 6: 0}  # TODO neutral

    initController = LatentWalkerController(0, doLoop=False)

    latentWalkers = [
        initController,
        LatentWalkerController(0, doLoop=False),
        LatentWalkerController(0, doLoop=False),
    ]
    latentIdxs = list(range(len(latentWalkers)))

    pointMapQueue: Queue = multiprocessing.Queue()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    camera_thread = multiprocessing.Process(
        target=emotions.doLoop, args=(latentIdxs, pointMapQueue), daemon=True
    )
    camera_thread.start()

    targetPM: list[PointDatum] = []
    pm: list[PointDatum] = []

    t = time()
    while True:
        cv2.waitKey(1)
        while pointMapQueue.qsize() > 1:
            targetPM = pointMapQueue.get()

        if len(targetPM) > 0:
            pm = smoothPoints(targetPM, pm)
            localLatentWalkers = [
                latentWalkers[point_datum.datum] for point_datum in pm
            ]

            for point_datum in pm:
                latentWalkers[point_datum.datum].targetClass = emotionsMap[
                    point_datum.emotion_label
                ]
            images = [lw.getImage() for lw in localLatentWalkers]

            hasNone = True
            while hasNone:
                hasNone = False
                for i in range(len(images)):
                    if images[i] is None:
                        hasNone = True
                        images[i] = localLatentWalkers[i].getImage()
                if hasNone:
                    sleep(0.001)

            scales = [
                (point_datum.point.w / 256) / 0.5 if point_datum.point.w > 0 else 1
                for point_datum in pm
            ]
            scales = [2 if s > 2 else s for s in scales]
            scales = [int(256 / s / 2) if s > 1 else 128 for s in scales]

            images = [
                cv2.resize(im[128 - s : 127 + s, 128 - s : 127 + s], (1024, 1024))
                for im, s in zip(images, scales)
            ]
            finalImage = points_to_voronoi(
                images,
                np.array(
                    [
                        [point_datum.point.normalized_x, point_datum.point.normalized_y]
                        for point_datum in pm
                    ]
                ),
                renderDots=False,
            )
            dt = time() - t
            if dt == 0:
                dt = 0.0001
            # print(f"camera FPS {1/dt}")
            # cv2.putText(
            #     finalImage,
            #     str(int(1 / dt)),
            #     (20, 20),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (255, 255, 255),
            #     2,
            # )

            cv2.imshow(
                "vornoi",
                finalImage,
            )
            t = time()
