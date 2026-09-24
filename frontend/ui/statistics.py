from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from database.db import Database
from ui.components import (
    TableColumn, data_table, kpi_card, page_header, section_header, status_badge,
)


def render(db: Database) -> None:
    page_header(
        "Báo cáo thống kê",
        "Theo dõi xu hướng hiện diện và tỷ lệ đúng giờ của nhân sự",
    )
    rows = db.list_attendance()
    if not rows:
        st.info("Chưa có dữ liệu điểm danh để lập báo cáo.")
        return

    frame = pd.DataFrame(rows)
    period = st.segmented_control(
        "Khoảng thời gian",
        ["7 ngày", "30 ngày", "Toàn bộ"],
        default="30 ngày",
        selection_mode="single",
    )
    if period != "Toàn bộ":
        days = 7 if period == "7 ngày" else 30
        frame = frame[frame["date"] >= (date.today() - timedelta(days=days - 1)).isoformat()]
    if frame.empty:
        st.info("Không có dữ liệu điểm danh trong khoảng thời gian đã chọn.")
        return

    frame["_present"] = (frame["check_in"].notna() & (frame["status"] != "ABSENT")).astype(int)
    daily = frame.groupby("date").agg(**{
        "Có mặt": ("_present", "sum"),
        "Đi muộn": ("status", lambda x: (x == "LATE").sum()),
        "Vắng mặt": ("status", lambda x: (x == "ABSENT").sum()),
        "Chờ chấm công": ("status", lambda x: (x == "WAITING").sum()),
    })
    daily["Tỷ lệ đúng giờ (%)"] = (
        (daily["Có mặt"] - daily["Đi muộn"]) / daily["Có mặt"].replace(0, float("nan")) * 100
    ).fillna(0).round(1)

    on_time = int((frame["status"] == "ON_TIME").sum())

    # KPI cards – Stitch colors
    present_count = int(frame["_present"].sum())
    late_count = int((frame["status"] == "LATE").sum())
    absent_count = int((frame["status"] == "ABSENT").sum())
    for column, card in zip(st.columns(4), (
        ("Đã tới văn phòng", present_count, "Có CHECK-IN",
         "fact_check", "#004ac6", "#dce9ff"),
        ("Lượt đúng giờ", on_time,
         f"{on_time / present_count * 100:.1f}% đã tới" if present_count else "Chưa có check-in",
         "check_circle", "#006c49", "rgba(108,248,187,0.30)"),
        ("Lượt đi muộn", late_count, "Cần theo dõi",
         "alarm_on", "#784b00", "#ffddb8"),
        ("Vắng mặt", absent_count, "Lịch ON, quá hạn chưa tới",
         "person_off", "#ba1a1a", "#ffdad6"),
    )):
        with column:
            kpi_card(*card)

    st.write("")
    left, right = st.columns([1.6, 1], gap="large")
    plot = daily.reset_index().rename(columns={"date": "Ngày"})
    plot["Ngày"] = pd.to_datetime(plot["Ngày"])

    # Stitch chart: primary=#004ac6
    with left:
        section_header("Lượt có mặt theo ngày")
        bars = (
            alt.Chart(plot)
            .mark_bar(color="#004ac6", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Ngày:T", title=None,
                         axis=alt.Axis(format="%d/%m", labelFontSize=11, labelColor="#434655")),
                y=alt.Y("Có mặt:Q", title="Nhân viên",
                         axis=alt.Axis(labelFontSize=11, labelColor="#434655")),
                tooltip=[alt.Tooltip("Ngày:T", format="%d/%m/%Y"), "Có mặt:Q", "Đi muộn:Q", "Vắng mặt:Q"],
            )
            .properties(height=300, background="transparent")
        )
        st.altair_chart(bars, width="stretch")

    # Stitch secondary green
    with right:
        section_header("Tỷ lệ đúng giờ")
        line = (
            alt.Chart(plot)
            .mark_area(
                line={"color": "#006c49", "strokeWidth": 2},
                color="rgba(108,248,187,0.25)",
            )
            .encode(
                x=alt.X("Ngày:T", title=None,
                         axis=alt.Axis(format="%d/%m", labelFontSize=11, labelColor="#434655")),
                y=alt.Y("Tỷ lệ đúng giờ (%):Q",
                         scale=alt.Scale(domain=[0, 100]),
                         axis=alt.Axis(labelFontSize=11, labelColor="#434655")),
                tooltip=[alt.Tooltip("Ngày:T", format="%d/%m/%Y"), "Tỷ lệ đúng giờ (%):Q"],
            )
            .properties(height=300, background="transparent")
        )
        st.altair_chart(line, width="stretch")

    section_header("Dữ liệu tổng hợp theo ngày")
    data_table(
        [
            TableColumn("date", "Ngày", 130),
            TableColumn("present", "Có mặt", 110, "center"),
            TableColumn("late", "Đi muộn", 110, "center"),
            TableColumn("absent", "Vắng mặt", 110, "center"),
            TableColumn("rate", "Tỷ lệ đúng giờ", 150, "center"),
        ],
        [{
            "date": row["Ngày"].strftime("%d/%m/%Y"),
            "present": int(row["Có mặt"]),
            "late": status_badge(str(int(row["Đi muộn"])), "warning"),
            "absent": status_badge(str(int(row["Vắng mặt"])), "danger"),
            "rate": status_badge(
                f'{row["Tỷ lệ đúng giờ (%)"]:.1f}%',
                "success" if row["Tỷ lệ đúng giờ (%)"] >= 80 else "warning",
            ),
        } for _, row in plot.iterrows()],
        compact=True,
    )
