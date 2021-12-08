from datetime import datetime
from time import sleep, time
import cv2
import numpy as np
from numpy.random.mtrand import randint
import pandas as pd
import sys
import dnnlib
from visualizer import AsyncRenderer
from threading import Thread

derpImage = cv2.imread("images/Emotions/fear/4 Avril.jpg")


class LatentWalk:
    def __init__(self, pkl=None, speed=0.25):
        t = time()
        self._async_renderer = AsyncRenderer()
        self._async_renderer._is_async = True
        self.current_class_idx = 0
        self.target_class_idx = 1
        self.args = dnnlib.EasyDict(pkl=pkl)
        self.speed = speed
        # exit()
        self.step_y = 1
        self.prevTime = time()
        self.targetX = 1

        self.prevTargetX = 0

        self.walkPercent = 0.0

        self.generate()
        print(f"LatentWalk took {time() - t}s to load")

    def generate(self, dx=None, dy=None):
        # self.drag(dx, dy)
        self.walk()
        if self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            # print(result)
            if result is not None and "image" in result:
                return result.image

        return None

    def walk(self):

        self.args.w0_seeds = []

        targetSeed = (round(self.targetX)) & ((1 << 32) - 1)

        currentSeed = (round(self.prevTargetX)) & ((1 << 32) - 1)

        currentWeight = 1 - self.walkPercent

        targetWeight = 1 - currentWeight
        self.args.w0_seeds.append([targetSeed, targetWeight, self.current_class_idx])
        self.args.w0_seeds.append([currentSeed, currentWeight, self.target_class_idx])
        print(self.args.w0_seeds)

        if self.walkPercent < 1:
            self.walkPercent +=( time() - self.prevTime )* self.speed
        else:
            self.walkPercent = 1
        self.prevTime = time()

        # for ofs_x, ofs_y in [[0, 0], [self.target, 0], [0, self.target], [self.target, self.target]]:
        #     seed_x = np.floor(self.latent.x) + ofs_x
        #     seed_y = np.floor(self.latent.y) + ofs_y
        #     seed = (int(seed_x) + int(seed_y) * self.step_y) & ((1 << 32) - 1)
        #     weight = (1 - abs(self.latent.x - seed_x)) * (1 - abs(self.latent.y - seed_y))
        #     if weight > 0:
        #         self.args.w0_seeds.append([seed, weight])


class LatentWalkerController:
    def __init__(self, targetClass):
        self.id = randint(696969)
        self.latent_walk = LatentWalk(pkl="network-snapshot-001460.pkl")
        self.latent_walk.target_class_idx = targetClass
        walker_thread = Thread(target=self.renderLoop, args=(self.latent_walk,))
        walker_thread.start()

        looper_thread = Thread(target=self.doLoop, args=())
        looper_thread.start()
        self.targetClass = targetClass

    def doLoop(self):
        while True:
            # userInX = int(input())
            # userInY = int(input())
            sleep(0.016)
            if self.latent_walk.walkPercent >= 1:
                self.latent_walk.prevTargetX = self.latent_walk.targetX
                # self.latent_walk.prevTargetX = 0

                # self.latent_walk.targetX = 0
                self.latent_walk.targetX = self.get_image()

                self.latent_walk.walkPercent = 0
                self.latent_walk.current_class_idx = self.latent_walk.target_class_idx
                self.latent_walk.target_class_idx = self.targetClass
                # print(self.latent_walk.targetX, self.latent_walk.targetY)

    def get_image(self):
        return np.random.randint(69696969)

    def renderLoop(self, latent_walk):
        t = time()
        while True:
            image = latent_walk.generate()
            if image is not None:
                colorCorrectedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(colorCorrectedImage, (512, 512))
                cv2.imshow("image " + str(self.id), resized_image)
                # cv2.waitKey(2)
                dt = time() - t
                if dt == 0:
                    dt = 0.001
                # print(f"fps for {self.id=} {1/(dt)}")

            if cv2.waitKey(17) & 0xFF == ord("q"):
                exit()
            t = time()


if __name__ == "__main__":
    controller = LatentWalkerController(0)
    # controller2 = LatentWalkerController()
    # controller3 = LatentWalkerController()
    while True:
        value = input()
        if value == "q":
            exit()
        print(type(value))
        controller.targetClass = int(value)

        # value = input()
        # if value == "q":
        #     exit()
        # print(type(value))
        # controller2.latent_walk.class_idx = int(value)
