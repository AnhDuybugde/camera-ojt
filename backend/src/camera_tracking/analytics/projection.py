from __future__ import annotations

import cv2
import numpy as np

from camera_tracking.domain import Point


class FloorProjector:
    """Project image pixels onto metric floor coordinates using homography."""

    def __init__(self, image_points: list[Point], floor_points: list[Point]) -> None:
        if len(image_points) != len(floor_points) or len(image_points) < 4:
            raise ValueError("Calibration requires at least four matching point pairs")
        source = np.asarray(image_points, dtype=np.float32)
        destination = np.asarray(floor_points, dtype=np.float32)
        matrix, _ = cv2.findHomography(source, destination)
        if matrix is None:
            raise ValueError("Calibration points do not define a valid homography")
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)

    def project(self, point: Point) -> Point:
        source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(source, self.matrix)[0, 0]
        return (float(projected[0]), float(projected[1]))

    def unproject(self, point: Point) -> Point:
        source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(source, self.inverse_matrix)[0, 0]
        return (float(projected[0]), float(projected[1]))
