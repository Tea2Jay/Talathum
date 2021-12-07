from time import sleep
import cv2
import numpy as np
import pandas as pd
from threading import Thread
from real_time.LatentWalker import LatentWalk


class LatentWalkerController:
    def __init__(self):
        self.latent_walk = LatentWalk(pkl="models/network-snapshot-001460.pkl")
        self.class_idx = 0

        walker_thread = Thread(target=renderLoop, args=(self.latent_walk,))
        walker_thread.setDaemon(True)
        walker_thread.start()

        looper_thread = Thread(target=self.doLoop, args=())
        looper_thread.setDaemon(True)
        looper_thread.start()

    def doLoop(self):
        while True:
            sleep(0.01)
            if self.latent_walk.walkPercent >= 1:

                self.latent_walk.prevTargetX = self.latent_walk.targetX
                self.latent_walk.prevTargetY = self.latent_walk.targetY
                self.latent_walk.prevTargetC = self.latent_walk.targetC

                self.latent_walk.targetX = self.get_image()
                self.latent_walk.targetY = self.get_image()
                self.latent_walk.targetC = self.latent_walk.class_idx

                self.latent_walk.walkPercent = 0

    def get_image(self):
        return np.random.randint(69696969)


def renderLoop(latent_walk):
    while True:
        image = latent_walk.generate()
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit()
