from camera_tracking.domain import BoundingBox, Detection
from camera_tracking.detection.yolo import suppress_nested_detections


def test_nested_person_box_is_removed_but_side_by_side_boxes_remain() -> None:
    outer = Detection(BoundingBox(0, 0, 100, 200), 0.80)
    nested = Detection(BoundingBox(10, 20, 90, 180), 0.60)
    other = Detection(BoundingBox(120, 0, 220, 200), 0.70)

    result = suppress_nested_detections([outer, nested, other])

    assert result == [outer, other]
