class Point:
    def __init__(self) -> None:
        self.normalized_x: float = 0
        self.normalized_y: float = 0
        self.x: float = 0
        self.y: float = 0
        self.w: float = 0
        self.h: float = 0

    def __add__(self, other):
        p = Point()
        p.normalized_x = self.normalized_x + other.normalized_x
        p.normalized_y = self.normalized_y + other.normalized_y
        p.x = self.x + other.x
        p.y = self.y + other.y
        p.w = self.w + other.w
        p.h = self.h + other.h
        return p

    def __sub__(self, other):
        p = Point()
        p.normalized_x = self.normalized_x - other.normalized_x
        p.normalized_y = self.normalized_y - other.normalized_y
        p.x = self.x - other.x
        p.y = self.y - other.y
        p.w = self.w - other.w
        p.h = self.h - other.h
        return p

    def __mul__(self, other):
        p = Point()
        p.normalized_x = self.normalized_x * other
        p.normalized_y = self.normalized_y * other
        p.x = self.x * other
        p.y = self.y * other
        p.w = self.w * other
        p.h = self.h * other
        return p

    def __str__(self) -> str:
        return f"nx={self.normalized_x}, ny={self.normalized_y}, x={self.x}, y={self.y}, w={self.w}, h={self.h}"

    __repr__ = __str__


class PointDatum:
    def __init__(self):
        self.point = Point()
        self.datum = 0
        self.emotions_array = []
        self.emotion_label = 0
