class Point:
    def __init__(self) -> None:
        self.normalized_x: float = 0
        self.normalized_y: float = 0
        self.x: float = 0
        self.y: float = 0
        self.w: float = 0
        self.h: float = 0


class PointDatum:
    def __init__(self):
        self.point = Point()
        self.datum = 0
        self.emotions_array = []
        self.emotion_label = None
