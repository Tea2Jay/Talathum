from threading import Thread
from time import sleep, time, time_ns
import cv2
from realTimeLatentWalk import LatentWalkerController
import emotions
from voronoi_image_merge import points_to_voronoi
import numpy as np

if __name__ == "__main__":

    initController = LatentWalkerController(0, doLoop=False)

    latentWalkers = [
        initController,
        LatentWalkerController(0, doLoop=False),
        LatentWalkerController(0, doLoop=False),
    ]

    camera_thread = Thread(target=emotions.doLoop, args=(latentWalkers,))
    camera_thread.start()

    t = time()
    while True:
        cv2.waitKey(1)
        if len(emotions.pointMap) > 0:
            localLatentWalkers = [lw for _, lw, _ in emotions.pointMap]
            images = [lw.getImage() for lw in localLatentWalkers]

            hasNone = True
            while hasNone:
                hasNone = False
                sleep(0.001)
                for i in range(len(images)):
                    if images[i] is None:
                        hasNone = True
                        images[i] = localLatentWalkers[i].getImage()
            images = [cv2.resize(im, (512, 512)) for im in images]
            finalImage = points_to_voronoi(
                images,
                np.array([point[0:2] for point, _, _ in emotions.pointMap]),
                renderDots=True,
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
