from ui import tracking_backend_view as live_attendance
from ui.tracking_backend_view import _active_tracking_people, _attendance_overview


def test_live_table_only_keeps_active_global_ids() -> None:
    payload = {
        "people": [
            {"gid": 1, "tracking_state": "ACTIVE"},
            {"gid": 2, "tracking_state": "TEMP_LOST"},
            {"gid": 3, "tracking_state": "LONG_LOST"},
            {"gid": 4, "tracking_state": "UNRESOLVED"},
        ]
    }

    assert [person["gid"] for person in _active_tracking_people(payload)] == [1]


def test_live_table_excludes_retained_person_not_currently_visible() -> None:
    payload = {
        "people": [
            {"gid": 1, "tracking_state": "TEMP_LOST", "visible": True},
            {"gid": 2, "tracking_state": "ACTIVE", "visible": False},
        ]
    }

    assert [person["gid"] for person in _active_tracking_people(payload)] == [1]


def test_attendance_overview_includes_checked_and_not_checked_roster(monkeypatch) -> None:
    class FakeDate:
        @classmethod
        def today(cls):
            return cls()

        def isoformat(self):
            return "2026-09-24"

    monkeypatch.setattr(live_attendance, "date", FakeDate)
    payload = {
        "employee_roster": [
            {"person_id": "01", "person_name": "Anh A", "gallery_samples": 1},
            {"person_id": "02", "person_name": "Chị B", "gallery_samples": 4},
        ],
        "attendance_today": [{
            "date": "2026-09-24", "person_id": "02",
            "person_name": "Chị B", "attended": True,
            "check_in_at": "2026-09-24T08:00:00+07:00", "face_score": 0.8,
        }],
    }

    rows = _attendance_overview(payload)

    assert [row["person_id"] for row in rows] == ["02", "01"]
    assert rows[0]["checked_in"] is True
    assert rows[1]["checked_in"] is False


def test_tracking_camera_can_render_expanded_stream(monkeypatch) -> None:
    rendered: list[tuple[str, bool]] = []

    def fake_markdown(body: str, *, unsafe_allow_html: bool = False) -> None:
        rendered.append((body, unsafe_allow_html))

    monkeypatch.setattr(live_attendance.st, "markdown", fake_markdown)

    live_attendance._render_tracking_camera(
        "http://127.0.0.1:8765/", "A", "/cam_a.mjpg", expanded=True
    )

    assert len(rendered) == 1
    html, unsafe = rendered[0]
    assert unsafe is True
    assert 'src="http://127.0.0.1:8765/cam_a.mjpg"' in html
    assert 'target="_blank"' in html
    assert "Camera A · chế độ phóng to" in html


def test_tracking_camera_escapes_stream_attributes(monkeypatch) -> None:
    rendered: list[str] = []
    monkeypatch.setattr(
        live_attendance.st,
        "markdown",
        lambda body, **_kwargs: rendered.append(body),
    )

    live_attendance._render_tracking_camera(
        'http://camera.invalid/\" onerror=\"alert(1)',
        '<A>',
        'cam_a.mjpg',
    )

    assert "&quot; onerror=&quot;alert(1)" in rendered[0]
    assert "Camera &lt;A&gt;" in rendered[0]
