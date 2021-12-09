import multiprocessing
from threading import Thread
from time import sleep, time, time_ns
from typing import List
import cv2
from realTimeLatentWalk import LatentWalkerController
import emotions
from voronoi_image_merge import points_to_voronoi
import numpy as np

if __name__ == "__main__":

    emotionsMap = {
        0:3,
        1:0,
        2:4,
        3:1,
        4:4,#TODO neutral
        5:2,
        6:0
    }

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

    pm = []

    t = time()
    while True:
        cv2.waitKey(1)
        while pointMapQueue.qsize() > 1:
            pm = pointMapQueue.get()

        if len(pm) > 0:
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

            images = [cv2.resize(im, (512, 512)) for im in images]
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
