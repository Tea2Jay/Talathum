import multiprocessing
from multiprocessing.queues import Queue
from sys import maxsize
from threading import Thread
from time import sleep, time, time_ns
from typing import List, Optional
import cv2
from realTimeLatentWalk import LatentWalkerController
import emotions
from voronoi_image_merge import points_to_voronoi
import numpy as np
from models import PointDatum, Point
import psutil


def smooth(
    prev: Optional[PointDatum], target: PointDatum, current: Optional[PointDatum]
) -> Point:
    if prev is None:
        return target.point

    if current is None:
        return target.point
    t = time()
    deltaTime = target.time - prev.time
    currentTime = t - target.time
    progress = currentTime / deltaTime
    progress -= 1

    if progress < 0:
        progress = 1
    if progress > 1:
        progress = 1

    expected_pos = target.point + (target.point - prev.point) * progress

    res = current.point + (expected_pos - current.point) * 0.5
    res.clip_normalized()

    return res


def smoothPoints(
    prev: list[PointDatum], target: list[PointDatum], current: list[PointDatum]
):
    newPM: list[PointDatum] = list.copy(target)

    for i, point_datum in enumerate(newPM):
        prev_point_datum = None
        current_point_datum = None

        for point_datum2 in current:
            if point_datum.datum == point_datum2.datum:
                current_point_datum = point_datum2
                break
        for point_datum2 in prev:
            if point_datum.datum == point_datum2.datum:
                prev_point_datum = point_datum2
                break
        pd = PointDatum()
        pd.copyFrom(newPM[i])
        pd.point = smooth(prev_point_datum, point_datum, current_point_datum)
        newPM[i] = pd
    return newPM


if __name__ == "__main__":

    cv2.namedWindow("vornoi", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("vornoi", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
    p = psutil.Process(camera_thread.pid)
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    camera_thread.start()

    prevPM: list[PointDatum] = []
    targetPM: list[PointDatum] = []
    pm: list[PointDatum] = []

    while True:
        cv2.waitKey(1)
        if pointMapQueue.qsize() > 1:
            prevPM = targetPM

        while pointMapQueue.qsize() > 1:
            targetPM = pointMapQueue.get()

        if len(targetPM) > 0:
            pm = smoothPoints(prevPM, targetPM, pm)
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

            x, y, w, h = cv2.getWindowImageRect("vornoi")

            w = int(w / 2)
            h = int(h / 2)
            minSize = int(max(min(w, h) / 1.5, 16))
            maxsize = int(max(max(w, h) / 1.5, 16))
            delta = maxsize - minSize

            scaledImages: list = []
            for (im, s) in zip(images, scales):
                if s < 1:
                    s = 1
                size = int(minSize + delta * (s - 1))
                scaledImages.append(
                    cv2.resize(
                        im,
                        (
                            size,
                            size,
                        ),
                    )
                )

            finalImage = points_to_voronoi(
                scaledImages,
                np.array(
                    [
                        [point_datum.point.normalized_x, point_datum.point.normalized_y]
                        for point_datum in pm
                    ]
                ),
                output_image_size=(h, w),
                renderDots=False,
            )

            cv2.imshow(
                "vornoi",
                finalImage,
            )
