from time import sleep, time
import cv2
import numpy as np
from numpy.random.mtrand import randint
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
            if result is not None and "image" in result:
                return result.image

        return None

    def walk(self):
        self.walkPercent += (time() - self.prevTime) * self.speed
        if self.walkPercent >= 1:
            self.walkPercent = 1
        self.prevTime = time()

        self.args.w0_seeds = []

        targetSeed = (round(self.targetX)) & ((1 << 32) - 1)

        currentSeed = (round(self.prevTargetX)) & ((1 << 32) - 1)

        currentWeight = 1 - self.walkPercent

        targetWeight = 1 - currentWeight
        self.args.w0_seeds.append([targetSeed, targetWeight, self.current_class_idx])
        self.args.w0_seeds.append([currentSeed, currentWeight, self.target_class_idx])


class LatentWalkerController:
    def __init__(self, targetClass, doLoop=False):
        self.id = randint(696969)
        self.latent_walk = LatentWalk(pkl="network-snapshot-001460.pkl")
        self.latent_walk.target_class_idx = targetClass
        if doLoop:
            walker_thread = Thread(target=self.renderLoop, args=())
            walker_thread.start()

        self.targetClass = targetClass

    def getImageSeed(self):
        return np.random.randint(69696969)

    def getImage(self):
        image = self.latent_walk.generate()

        if self.latent_walk.walkPercent >= 1:
            self.latent_walk.prevTargetX = self.latent_walk.targetX
            self.latent_walk.targetX = self.getImageSeed()
            self.latent_walk.walkPercent = 0
            self.latent_walk.current_class_idx = self.latent_walk.target_class_idx
            self.latent_walk.target_class_idx = self.targetClass

        if image is not None:
            colorCorrectedImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # return np.full((256, 256, 3), self.id % 255, dtype=np.uint8)
            return colorCorrectedImage

    def renderLoop(self):
        t = time()
        while True:
            im = self.getImage()
            if im is not None:
                cv2.imshow("image " + str(self.id), im)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                exit()


if __name__ == "__main__":
    controller = LatentWalkerController(0, doLoop=True)
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
