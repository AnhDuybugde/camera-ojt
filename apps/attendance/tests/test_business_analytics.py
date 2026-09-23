from __future__ import annotations

from datetime import date

from spatial.business_analytics import attendance_flow, attendance_frame, gate_flow, occupancy_timeline, peak


def _rows() -> list[dict]:
    return [
        {"employee_id": "NV001", "employee_name": "A", "department": "AI", "date": "2026-09-16",
         "check_in": "2026-09-16T08:07:00", "check_out": "2026-09-16T17:31:00"},
        {"employee_id": "NV002", "employee_name": "B", "department": "Dev", "date": "2026-09-16",
         "check_in": "2026-09-16T08:12:00", "check_out": None},
        {"employee_id": "NV003", "employee_name": "C", "department": "AI", "date": "2026-09-16",
         "check_in": "2026-09-16T08:22:00", "check_out": "2026-09-16T17:35:00"},
    ]


def test_attendance_flow_and_peak() -> None:
    flow = attendance_flow(attendance_frame(_rows()), 15)
    assert peak(flow, "Check-in") == ("08:00", 2)
    assert peak(flow, "Check-out") == ("17:30", 2)


def test_attendance_department_filter_and_occupancy() -> None:
    frame = attendance_frame(_rows(), ["AI"])
    timeline = occupancy_timeline(frame, date(2026, 9, 16), 15)
    assert len(frame) == 2
    assert int(timeline["Hiện diện"].max()) == 2
    assert int(timeline["Hiện diện"].iloc[-1]) == 0


def test_gate_flow_uses_entrance_transitions_only() -> None:
    events = [
        {"event_time": "2026-09-16T08:03:00", "from_zone": "entrance", "to_zone": "center"},
        {"event_time": "2026-09-16T08:05:00", "from_zone": "center", "to_zone": "workspace"},
        {"event_time": "2026-09-16T17:35:00", "from_zone": "workspace", "to_zone": "entrance"},
    ]
    flow = gate_flow(events, 15)
    assert flow[["Loại", "Số lượt"]].to_dict("records") == [
        {"Loại": "Entry camera", "Số lượt": 1},
        {"Loại": "Exit camera", "Số lượt": 1},
    ]
