from datetime import datetime
from time import sleep
import cv2
import numpy as np
import pandas as pd

import dnnlib
from visualizer import AsyncRenderer
from threading import Thread


class LatentWalk:
    def __init__(self, pkl=None, x=0, y=0, speed=0.25, step_y=100):
        self._async_renderer = AsyncRenderer()
        self.args = dnnlib.EasyDict(pkl=pkl)
        self.latent = dnnlib.EasyDict(x=x, y=y, speed=speed)
        self.step_y = 1
        self.prevTime = datetime.now()
        self.targetX = 1
        self.targetY = 1

        self.prevTargetX = 0
        self.prevTargetY = 0

        self.walkPercent = 0.0

    def generate(self, dx=None, dy=None):
        # self.drag(dx, dy)
        self.walk()
        if self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            if result is not None and "image" in result:
                return result.image

        self.prevTime = datetime.now()

        return None

    def walk(self):

        self.args.w0_seeds = []

        targetSeed = (int(self.targetX) + int(self.targetY) * self.step_y) & (
            (1 << 32) - 1
        )

        currentSeed = (
            round(self.prevTargetX) + round(self.prevTargetY) * self.step_y
        ) & ((1 << 32) - 1)

        currentWeight = 1 - self.walkPercent

        targetWeight = 1 - currentWeight

        self.args.w0_seeds.append([targetSeed, targetWeight])
        self.args.w0_seeds.append([currentSeed, currentWeight])

        self.latent.x = (
            self.targetX - self.prevTargetX
        ) * self.walkPercent + self.prevTargetX
        self.latent.y = (
            self.targetY - self.prevTargetY
        ) * self.walkPercent + self.prevTargetY

        if self.walkPercent < 1:
            self.walkPercent += 0.01
        else:
            self.walkPercent = 1

        # for ofs_x, ofs_y in [[0, 0], [self.target, 0], [0, self.target], [self.target, self.target]]:
        #     seed_x = np.floor(self.latent.x) + ofs_x
        #     seed_y = np.floor(self.latent.y) + ofs_y
        #     seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
        #     weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
        #     if weight > 0:
        #         self.args.w0_seeds.append([seed, weight])


class LatentWalkerController:
    def __init__(self):
        self.latent_walk = LatentWalk(pkl="network-snapshot-000660.pkl")
        self.target_emotion = "h"
        self.emotion_dictionary = {}
        self.read_emotions("emotion distribution.csv")

        walker_thread = Thread(target=renderLoop, args=(self.latent_walk,))
        walker_thread.start()

        looper_thread = Thread(target=self.doLoop, args=())
        looper_thread.start()

    def read_emotions(self, path):
        csv = pd.read_csv(path)
        for row in csv.iloc:
            if not row[1] in self.emotion_dictionary: 
                self.emotion_dictionary[row[1]] = []

            self.emotion_dictionary[row[1]].append(row[0])
        print(self.emotion_dictionary)

    def doLoop(self):
        while True:
            # userInX = int(input())
            # userInY = int(input())
            sleep(0.01)
            if self.latent_walk.walkPercent >= 1:
                self.latent_walk.prevTargetX = 0
                self.latent_walk.prevTargetY = self.latent_walk.targetY

                self.latent_walk.targetX = 0
                self.latent_walk.targetY = self.get_image()

                self.latent_walk.walkPercent = 0
                print(self.latent_walk.targetX, self.latent_walk.targetY)

    def get_image(self):
        emotionList = self.emotion_dictionary[self.target_emotion]
        return emotionList[np.random.randint(0, len(emotionList))]


def renderLoop(latent_walk):
    while True:
        image = latent_walk.generate()
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
        cv2.imshow("image", image)
        # cv2.waitKey(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    controller = LatentWalkerController()
    while True:
        input()
        controller.target_emotion = 's'
        input()
        controller.target_emotion = 'a' 
        input()
        controller.target_emotion = 'h'
    
