"""Small, reusable presentation helpers – Stitch design system.

Icon convention
---------------
- Streamlit-native buttons / inputs : use :material/icon_name: syntax (built-in).
- Custom HTML markup                 : use Bootstrap Icons class (bi bi-xxx) – loaded via CDN.

Stitch color tokens used here match the exact values in assets/style.css.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from typing import Any, Mapping, Sequence

import streamlit as st


@dataclass(frozen=True, slots=True)
class TableColumn:
    key: str
    label: str
    min_width: int = 110
    align: str = "left"


@dataclass(frozen=True, slots=True)
class EmployeeCell:
    name: str
    employee_id: str
    subtitle: str = ""
    show_id: bool = True


@dataclass(frozen=True, slots=True)
class StatusBadge:
    label: str
    tone: str = "neutral"

# ── Bootstrap Icons mapping ──────────────────────────────────────────────────
# Map logical icon names → Bootstrap Icons class suffix
_BI: dict[str, str] = {
    "groups":            "people-fill",
    "how_to_reg":        "person-check-fill",
    "check_circle":      "check-circle-fill",
    "schedule":          "clock-history",
    "alarm_on":          "alarm-fill",
    "person_off":        "person-x-fill",
    "face":              "person-bounding-box",
    "corporate_fare":    "building-fill",
    "fact_check":        "clipboard2-check-fill",
    "task_alt":          "check-circle-fill",
    "event_available":   "calendar-check-fill",
    "calendar_month":    "calendar-week-fill",
    "home_work":         "house-door-fill",
    "sync":              "arrow-repeat",
    "settings":          "gear-fill",
    "grid_view":         "grid-fill",
    "database":          "database-fill",
    "bar_chart":         "bar-chart-fill",
    "insights":          "graph-up-arrow",
    "trending_up":       "graph-up",
    "verified":          "patch-check-fill",
    "videocam":          "camera-video-fill",
    "videocam_off":      "camera-video-off-fill",
    "search":            "search",
    "download":          "download",
    "wifi":              "wifi",
    "warning":           "exclamation-triangle-fill",
    "check":             "check-lg",
    "login":             "box-arrow-in-right",
    "link":              "link-45deg",
}


def _bi(name: str, size: str = "") -> str:
    """Return an HTML <i> element using Bootstrap Icons."""
    cls = _BI.get(name, name.replace("_", "-"))
    sz = f" {size}" if size else ""
    return f'<i class="bi bi-{cls}{sz}" aria-hidden="true"></i>'


# ── Page header ──────────────────────────────────────────────────────────────

def page_header(title: str, description: str, *, status: str | None = None) -> None:
    status_html = ""
    if status:
        status_html = (
            f'<div class="stitch-status-chip">'
            f'  <span class="stitch-status-dot-anim"></span>'
            f'  {escape(status)}'
            f'</div>'
        )
    st.markdown(
        f'<div class="stitch-page-head">'
        f'  <div>'
        f'    <div class="stitch-breadcrumb">'
        f'      <span>Trang chủ</span>'
        f'      <span class="stitch-breadcrumb-sep">/</span>'
        f'      <span class="stitch-breadcrumb-active">{escape(title)}</span>'
        f'    </div>'
        f'    <div class="stitch-page-title-row">'
        f'      <h1 class="stitch-page-title">{escape(title)}</h1>'
        f'    </div>'
        f'    <div style="font-size:12px;color:var(--st-on-surface-variant);margin-top:4px;overflow-wrap:anywhere;">'
        f'      {escape(description)}'
        f'    </div>'
        f'  </div>'
        f'  {status_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── KPI card ─────────────────────────────────────────────────────────────────

def kpi_card(
    label: str,
    value: str | int,
    note: str,
    icon: str,
    color: str,       # icon color (hex)
    soft: str,        # icon bg / accent-soft (hex)
) -> None:
    icon_html = _bi(icon)
    # Determine value color from accent color
    value_color = color
    # Map Stitch semantic colors
    _color_map = {
        "#10b981": "var(--st-secondary)",
        "#f59e0b": "var(--st-tertiary)",
        "#ef4444": "var(--st-error)",
        "#2563eb": "var(--st-on-surface)",
        "#8b5cf6": "#6d28d9",
    }
    v_color = _color_map.get(color.lower(), "var(--st-on-surface)")
    # Only late/absent get colored value
    if color.lower() in ("#f59e0b", "#ef4444"):
        val_style = f"color:{v_color};"
    else:
        val_style = "color:var(--st-on-surface);"

    st.markdown(
        f'<div class="stitch-kpi" style="--kpi-icon-color:{color};--kpi-icon-bg:{soft};--kpi-accent-soft:{soft}50;">'
        f'  <div>'
        f'    <div class="stitch-kpi-top">'
        f'      <span class="stitch-kpi-label">{escape(label)}</span>'
        f'      <span class="stitch-kpi-icon">{icon_html}</span>'
        f'    </div>'
        f'    <div class="stitch-kpi-value" style="{val_style}">{escape(str(value))}</div>'
        f'    <div class="stitch-kpi-sub">{escape(note)}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Section header ────────────────────────────────────────────────────────────

def section_header(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="stitch-section-head">'
        f'  <div class="stitch-section-title">{escape(title)}</div>'
        f'  <div class="stitch-section-note">{escape(note)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def employee_cell(
    name: str, employee_id: str, subtitle: str = "", *, show_id: bool = True
) -> EmployeeCell:
    return EmployeeCell(name, employee_id, subtitle, show_id)


def status_badge(label: str, tone: str = "neutral") -> StatusBadge:
    return StatusBadge(label, tone)


def data_table(
    columns: Sequence[TableColumn],
    rows: Sequence[Mapping[str, Any]],
    *,
    empty_message: str = "Chưa có dữ liệu.",
    compact: bool = False,
    max_height: int | None = None,
) -> None:
    """Render the Work Schedule table language for read-only data grids."""
    if not rows:
        st.markdown(
            '<div class="ds-table-empty"><i class="bi bi-inbox"></i>'
            f'<b>{escape(empty_message)}</b></div>',
            unsafe_allow_html=True,
        )
        return

    minimum = sum(column.min_width for column in columns)
    head = "".join(
        f'<th style="width:{column.min_width}px;min-width:{column.min_width}px;text-align:{column.align}">'
        f'{escape(column.label)}</th>' for column in columns
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td style="width:{column.min_width}px;min-width:{column.min_width}px;text-align:{column.align}">'
            f'{_table_value(row.get(column.key))}</td>' for column in columns
        ) + "</tr>"
        for row in rows
    )
    height_style = f"max-height:{max_height}px;" if max_height else ""
    density = " ds-table-compact" if compact else ""
    st.markdown(
        f'<div class="ds-table{density}"><div class="ds-table-scroll" '
        f'style="{height_style}"><table style="min-width:{minimum}px">'
        f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody>'
        '</table></div></div>',
        unsafe_allow_html=True,
    )


def _table_value(value: Any) -> str:
    if isinstance(value, EmployeeCell):
        words = [part for part in value.name.split() if part]
        initials = "".join(part[0].upper() for part in words[-2:]) or "NV"
        subtitle = value.subtitle or ""
        employee_id = (
            f'<code>{escape(value.employee_id)}</code>' if value.show_id else ""
        )
        return (
            '<div class="ds-employee-cell">'
            f'<span class="ds-avatar">{escape(initials)}</span><span class="ds-person">'
            f'<b class="ds-person-name" title="{escape(value.name)}">{escape(value.name)}</b>'
            f'<small>{escape(subtitle)}</small></span>'
            f'{employee_id}</div>'
        )
    if isinstance(value, StatusBadge):
        tone = value.tone if value.tone in {
            "success", "info", "warning", "danger", "neutral", "primary"
        } else "neutral"
        return f'<span class="ds-badge ds-badge-{tone}">{escape(value.label)}</span>'
    if isinstance(value, (list, tuple)):
        return '<div class="ds-badge-list">' + "".join(_table_value(item) for item in value) + "</div>"
    text = "—" if value is None or value == "" else str(value)
    return f'<span class="ds-cell-text" title="{escape(text)}">{escape(text)}</span>'


# ── Vietnamese date ───────────────────────────────────────────────────────────

def vietnamese_date(value: date | None = None) -> str:
    value = value or date.today()
    weekdays = ("Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật")
    return f"{weekdays[value.weekday()]}, {value:%d/%m/%Y}"


# ── Message translator ────────────────────────────────────────────────────────

def translate_message(message: str) -> str:
    """Translate known service messages without changing backend contracts."""
    replacements = {
        "Full name is required.":
            "Họ và tên là thông tin bắt buộc.",
        "Employee ID must contain 2-32 letters, numbers, '_' or '-'.":
            "Mã nhân viên phải có 2–32 ký tự chữ, số, dấu gạch dưới hoặc gạch ngang.",
        "Employee does not exist.":
            "Nhân viên không tồn tại.",
        "Employee already has a face embedding. Delete/replace it first.":
            "Nhân viên đã có dữ liệu khuôn mặt. Hãy xóa hoặc thay thế dữ liệu cũ trước.",
        "Google Sheets is not configured; records remain pending.":
            "Google Sheets chưa được cấu hình; các bản ghi vẫn nằm trong hàng đợi.",
        "No pending records.":
            "Không có bản ghi đang chờ.",
        "Waiting for camera...":
            "Đang chờ camera…",
        "Camera connection lost; reconnecting...":
            "Mất kết nối camera; hệ thống đang kết nối lại…",
    }
    if message in replacements:
        return replacements[message]
    if message.startswith("Cannot open camera:"):
        return "Không thể mở camera đã cấu hình."
    if message.startswith("Recognition error:"):
        return "Lỗi xử lý nhận diện: " + message.split(":", 1)[1].strip()
    if message.startswith("Captured only "):
        return (message
                .replace("Captured only ", "Chỉ thu được ")
                .replace(" valid samples before timeout.", " mẫu hợp lệ trước khi hết thời gian."))
    if message.startswith("Move closer: the face is too small."):
        return "Hãy tiến gần hơn: khuôn mặt đang quá nhỏ."
    if message.startswith("Face detected, but image is blurry"):
        return "Đã thấy khuôn mặt nhưng hình ảnh còn mờ. Hãy giữ yên và cải thiện ánh sáng."
    if message.startswith("Detected ") and "faces" in message:
        return "Phát hiện nhiều khuôn mặt. Chỉ một người được xuất hiện trong khung hình."
    return message
