"""Pure business analytics derived from attendance and spatial events."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

import pandas as pd


FLOW_COLUMNS = ["Mốc", "Loại", "Số lượt", "Phút"]


def attendance_frame(
    rows: Sequence[dict], departments: Iterable[str] = (),
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=[
            "employee_id", "employee_name", "department", "date", "check_in", "check_out",
        ])
    for column in ("check_in", "check_out"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["department"] = frame["department"].fillna("").replace("", "Chưa phân loại")
    selected = set(departments)
    if selected:
        frame = frame[frame["department"].isin(selected)]
    return frame.copy()


def attendance_flow(frame: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    """Aggregate check-in/out events by clock bucket across selected days."""
    events: list[dict] = []
    for column, label in (("check_in", "Check-in"), ("check_out", "Check-out")):
        if column not in frame:
            continue
        for value in frame[column].dropna():
            minute = (value.hour * 60 + value.minute) // bucket_minutes * bucket_minutes
            events.append({"Mốc": _clock_label(minute), "Loại": label, "Phút": minute})
    if not events:
        return pd.DataFrame(columns=FLOW_COLUMNS)
    result = (
        pd.DataFrame(events)
        .groupby(["Mốc", "Loại", "Phút"], as_index=False)
        .size()
        .rename(columns={"size": "Số lượt"})
        .sort_values(["Phút", "Loại"])
    )
    return result[FLOW_COLUMNS]


def peak(flow: pd.DataFrame, event_type: str) -> tuple[str, int]:
    selected = flow[flow["Loại"] == event_type] if not flow.empty else flow
    if selected.empty:
        return "—", 0
    row = selected.loc[selected["Số lượt"].idxmax()]
    return str(row["Mốc"]), int(row["Số lượt"])


def occupancy_timeline(
    frame: pd.DataFrame, selected_day: date, bucket_minutes: int,
) -> pd.DataFrame:
    """Cumulative attendance occupancy for one day (check-in +1, check-out -1)."""
    if frame.empty:
        return pd.DataFrame(columns=["Mốc thời gian", "Hiện diện"])
    day_rows = frame[frame["date"].astype(str) == selected_day.isoformat()]
    events: list[tuple[pd.Timestamp, int]] = []
    for _, row in day_rows.iterrows():
        if pd.notna(row.get("check_in")):
            events.append((pd.Timestamp(row["check_in"]), 1))
        if pd.notna(row.get("check_out")):
            events.append((pd.Timestamp(row["check_out"]), -1))
    if not events:
        return pd.DataFrame(columns=["Mốc thời gian", "Hiện diện"])
    event_frame = pd.DataFrame(events, columns=["time", "delta"]).set_index("time")
    series = event_frame["delta"].resample(f"{bucket_minutes}min").sum().cumsum().clip(lower=0)
    return series.rename("Hiện diện").rename_axis("Mốc thời gian").reset_index()


def gate_flow(
    events: Sequence[dict], bucket_minutes: int,
    *, entrance_zone: str = "entrance",
    interior_zones: Iterable[str] = ("center", "workspace"),
) -> pd.DataFrame:
    """Classify zone transitions as physical entry or exit events."""
    interior = set(interior_zones)
    classified: list[dict] = []
    for event in events:
        from_zone, to_zone = event.get("from_zone"), event.get("to_zone")
        movement = None
        if from_zone == entrance_zone and to_zone in interior:
            movement = "Entry camera"
        elif to_zone == entrance_zone and from_zone in interior:
            movement = "Exit camera"
        if not movement:
            continue
        timestamp = pd.to_datetime(event.get("event_time"), errors="coerce")
        if pd.isna(timestamp):
            continue
        minute = (timestamp.hour * 60 + timestamp.minute) // bucket_minutes * bucket_minutes
        classified.append({
            "Mốc": _clock_label(minute), "Loại": movement, "Phút": minute,
            "event_time": timestamp, "employee_id": event.get("employee_id"),
            "employee_name": event.get("employee_name") or "Chưa xác định",
            "camera_id": event.get("camera_id") or "—",
        })
    if not classified:
        return pd.DataFrame(columns=[*FLOW_COLUMNS, "event_time", "employee_id", "employee_name", "camera_id"])
    raw = pd.DataFrame(classified)
    grouped = raw.groupby(["Mốc", "Loại", "Phút"], as_index=False).size().rename(columns={"size": "Số lượt"})
    return grouped.sort_values(["Phút", "Loại"])


def filter_spatial_events(
    rows: Sequence[dict], start_day: date, end_day: date,
) -> list[dict]:
    start = pd.Timestamp(start_day)
    end = pd.Timestamp(end_day) + pd.Timedelta(days=1)
    filtered = []
    for row in rows:
        timestamp = pd.to_datetime(row.get("event_time"), errors="coerce")
        if pd.isna(timestamp):
            continue
        comparable = timestamp.tz_localize(None) if timestamp.tzinfo else timestamp
        if start <= comparable < end:
            filtered.append(row)
    return filtered


def _clock_label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"
