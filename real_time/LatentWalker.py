import sys
import dnnlib
from visualizer import AsyncRenderer
from datetime import datetime


class LatentWalk:
    def __init__(self, pkl=None, x=0, y=0, speed=0.25, step_y=100):
        self._async_renderer = AsyncRenderer()
        self.args = dnnlib.EasyDict(pkl=pkl)
        self.latent = dnnlib.EasyDict(x=x, y=y, speed=speed)
        self.step_y = 1
        self.prevTime = datetime.now()
        self.targetX = 1
        self.targetY = 1
        self.targetC = 1

        self.prevTargetX = 0
        self.prevTargetY = 0
        self.prevTargetC = 0

        self.walkPercent = 0.0

    def generate(self, dx=None, dy=None):
        # self.drag(dx, dy)
        self.walk()
        if self.args.pkl is not None:
            self._async_renderer.set_args(**self.args)
            result = self._async_renderer.get_result()
            # print(result)
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

        self.args.class_idx = [self.targetC, self.prevTargetC]
        self.args.w0_seeds.append([targetSeed, targetWeight])
        self.args.w0_seeds.append([currentSeed, currentWeight])

        if self.walkPercent < 1:
            self.walkPercent += 0.01
        else:
            self.walkPercent = 1
