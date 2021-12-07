from time import time
import numpy as np
from numpy.lib.type_check import imag
from scipy.spatial import Voronoi
import cv2

from shapely.geometry import Polygon


def clip(arr, min, max):
    p1 = Polygon(arr)
    p2 = Polygon([(min, min), (min, max), (max, max), (max, min)])

    inter1 = p1.intersection(p2)
    cords = list(inter1.exterior.coords)

    return cords


def vorarr(images, regions, vertices, points, renderDots=False):

    res = np.zeros_like(images[0])
    for i, (region, image, point) in enumerate(zip(regions, images, points)):
        polygon = vertices[region]
        # ax.fill(
        #     *zip(*polygon),
        # )
        clippedPolygon = clip(vertices[region] * 256 + 256, 0, 512)
        clippedPolygon = np.array(clippedPolygon, dtype=np.int32)

        [maxX, maxY] = np.max(clippedPolygon, axis=0)
        [minX, minY] = np.min(clippedPolygon, axis=0)

        deltaX = maxX - minX
        deltaY = maxY - minY

        centerX = minX + deltaX / 2
        centerY = minY + deltaY / 2

        maxDelta = deltaX if deltaX > deltaY else deltaY

        mask = np.zeros(image.shape, dtype=image.dtype)
        mask = cv2.fillConvexPoly(mask, clippedPolygon, (1, 1, 1))

        maskedImage = np.multiply(image, mask)

        maskOut = 1 - mask

        maskedRes = np.multiply(res, maskOut)

        res = np.add(maskedImage, maskedRes)

        if renderDots:
            point = point * 256 + 256
            cv2.circle(
                res, (int(point[0]), int(point[1])), 5, (120, 90, 255), thickness=3
            )

    return res


def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


import time


def points_to_voronoi(images, points, renderDots=False):
    images = images[: points.shape[0]][:]
    points = np.copy(points)
    points = np.append(points, [[9999, 9999]], axis=0)
    points = np.append(points, [[-9999, 9999]], axis=0)
    points = np.append(points, [[-9999, -9999]], axis=0)
    points = np.append(points, [[9999, -9999]], axis=0)

    # compute Voronoi tesselation
    vor = Voronoi(points)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    # voronoi_finite_polygons_2d function from https://stackoverflow.com/a/20678647/425458

    # print(f"{regions=} {vertices=} {vertices=} ")
    arr = vorarr(images, regions, vertices, points, renderDots=renderDots)

    return arr


if __name__ == "__main__":
    np.random.seed(1234)
    points = np.random.rand(1, 2)
    speeds = np.random.rand(4, 2) - 0.5
    speeds /= 30
    image_paths = [
        "images/1024/3970-00.png",
        "images/1024/822737-00.png",
        "images/Emotions/Anger/Seated Nude.jpg",
        "images/Emotions/Anger/The Family.jpg",
    ]
    images = [cv2.resize(cv2.imread(path), (512, 512)) for path in image_paths]
    np.set_printoptions(suppress=True)

    prevTime = time.time()
    while True:
        timeD = time.time() - prevTime
        print(1 / (timeD if timeD > 0 else 0.001))
        prevTime = time.time()

        arr = points_to_voronoi(images, points, renderDots=True)
        # plot the numpy array
        cv2.imshow("", arr)
        if cv2.waitKey(1) != -1:
            break

        for idx, j in np.ndenumerate(points):
            points[idx] = j + speeds[idx]
            if points[idx] > 1:
                speeds[idx] *= -1
            elif points[idx] < -1:
                speeds[idx] *= -1
