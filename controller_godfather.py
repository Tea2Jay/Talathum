import multiprocessing
from threading import Thread
from time import sleep, time, time_ns
from typing import List
import cv2
from realTimeLatentWalk import LatentWalkerController
import emotions
from voronoi_image_merge import points_to_voronoi
import numpy as np


def lerp(point1, point2, factor):
    return [(p2 + ((p1 - p2) * factor)) for p1, p2 in zip(point1, point2)]


def smoothPoints(target, current):
    lerpFactor = 0.1

    newPM = list.copy(target)

    for i, [point, lwid, _] in enumerate(newPM):
        for point2, lwid2, _ in current:
            if lwid == lwid2:
                newPM[i][0] = lerp(point, point2, lerpFactor)
                print(f"{point=} {point2=} {newPM[i][0]=}")
                break
    return newPM


if __name__ == "__main__":

    emotionsMap = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2, 5: 2, 6: 0}  # TODO neutral

    initController = LatentWalkerController(0, doLoop=False)

    latentWalkers = [
        initController,
        LatentWalkerController(0, doLoop=False),
        LatentWalkerController(0, doLoop=False),
    ]
    latentIdxs = list(range(len(latentWalkers)))

    pointMapQueue = multiprocessing.Queue()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    camera_thread = multiprocessing.Process(
        target=emotions.doLoop, args=(latentIdxs, pointMapQueue), daemon=True
    )
    camera_thread.start()

    targetPM = []
    pm = []

    t = time()
    while True:
        cv2.waitKey(1)
        while pointMapQueue.qsize() > 1:
            targetPM = pointMapQueue.get()

        if len(targetPM) > 0:
            pm = smoothPoints(targetPM, pm)
            localLatentWalkers = [latentWalkers[lwid] for _, lwid, _ in pm]

            for _, lwid, emotionsData in pm:
                latentWalkers[lwid].targetClass = emotionsMap[emotionsData[1]]
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
                (point[4] / 256) / 0.3 if point[4] > 0 else 1 for point, _, _ in pm
            ]
            scales = [2 if s > 2 else s for s in scales]
            scales = [int(256 / s / 2) if s > 1 else 128 for s in scales]

            images = [
                cv2.resize(im[128 - s : 127 + s, 128 - s : 127 + s], (512, 512))
                for im, s in zip(images, scales)
            ]
            finalImage = points_to_voronoi(
                images,
                np.array([point[0:2] for point, _, _ in pm]),
                renderDots=False,
            )
            dt = time() - t
            if dt == 0:
                dt = 0.0001
            # print(f"camera FPS {1/dt}")
            cv2.putText(
                finalImage,
                str(int(1 / dt)),
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "vornoi",
                finalImage,
            )
            t = time()
