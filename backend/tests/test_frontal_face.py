"""is_frontal_face: mat nhin thang moi mo cong trigger tay."""
from camera_tracking.face.embeddings import is_frontal_face


# mat trai, mat phai, mui, mieng trai, mieng phai.
FRONT = ((30.0, 40.0), (70.0, 40.0), (50.0, 55.0), (35.0, 75.0), (65.0, 75.0))


def test_frontal_accepted() -> None:
    assert is_frontal_face(FRONT, 100.0) is True


def test_profile_rejected() -> None:
    # Mui lech gan het sang phai (nhin xeo).
    side = ((30.0, 40.0), (70.0, 40.0), (68.0, 55.0), (35.0, 75.0), (65.0, 75.0))
    assert is_frontal_face(side, 100.0) is False


def test_rolled_face_rejected() -> None:
    # 2 mat lech cao do qua nhieu (nghieng dau).
    rolled = ((30.0, 30.0), (70.0, 55.0), (50.0, 55.0), (35.0, 75.0), (65.0, 75.0))
    assert is_frontal_face(rolled, 100.0) is False


def test_bad_vertical_order_rejected() -> None:
    # Mui khong nam giua mat va mieng (kps loi / goc cuc doan).
    bad = ((30.0, 40.0), (70.0, 40.0), (50.0, 90.0), (35.0, 75.0), (65.0, 75.0))
    assert is_frontal_face(bad, 100.0) is False


def test_missing_or_tiny_kps_rejected() -> None:
    assert is_frontal_face(None, 100.0) is False
    assert is_frontal_face(FRONT[:4], 100.0) is False
    # Mat qua nho: 2 mat chiem < 20% rong mat.
    assert is_frontal_face(FRONT, 500.0) is False


def test_tolerance_configurable() -> None:
    near = ((30.0, 40.0), (70.0, 40.0), (56.0, 55.0), (35.0, 75.0), (65.0, 75.0))
    assert is_frontal_face(near, 100.0) is True
    assert is_frontal_face(near, 100.0, nose_tol=0.05) is False
