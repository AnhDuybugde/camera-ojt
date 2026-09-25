"""Read-only attendance styled like the weekly schedule."""
from datetime import date, datetime, timedelta
from html import escape
import streamlit as st
from auth.permissions import require_owner


def _time(value):
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime("%H:%M:%S")
    except ValueError:
        return escape(str(value))


def render(db, role):
    owner = st.session_state.get("employee_id")
    require_owner(role, owner or "", owner)
    employee = db.get_employee(owner) if owner else None
    if not employee:
        st.error("Không tìm thấy hồ sơ nhân viên liên kết với tài khoản.")
        return
    today = date.today()
    st.markdown('<div class="ws-page-head"><div><div class="ws-title-line"><h1>Chấm công của tôi</h1>'
                '<span class="ws-standard"><span></span>Cá nhân</span></div>'
                f'<p>{escape(employee["full_name"])} · {escape(owner)} — Theo dõi điểm danh hôm nay và lịch sử chấm công của bạn.</p></div></div>', unsafe_allow_html=True)
    records = db.list_attendance(today.isoformat(), today.isoformat(), employee_id=owner)
    record = records[0] if records else {}
    checked_in, checked_out = bool(record.get("check_in")), bool(record.get("check_out"))
    message = ("Đã ghi nhận CHECK-IN và CHECK-OUT hôm nay." if checked_in and checked_out
               else "Đã điểm danh CHECK-IN thành công hôm nay." if checked_in
               else "Chưa ghi nhận CHECK-IN hôm nay.")
    tone = "on" if checked_in else "empty"
    initials = ''.join(word[0] for word in employee["full_name"].split()[-2:]).upper()
    identity = (f'<div class="pa-person"><span class="pa-avatar">{escape(initials)}</span><div>'
                f'<b>{escape(employee["full_name"])}</b><small>{escape(employee.get("position") or "Nhân viên")}</small>'
                f'</div><span class="pa-code">{escape(owner)}</span></div>')
    def time_cell(value):
        color = "on" if value else "empty"
        return f'<span class="pa-time-pill ws-legend-{color}"><i class="bi bi-clock"></i> {_time(value)}</span>'
    st.markdown(f'<div class="pa-history"><div class="pa-history-heading"><b>Điểm danh hôm nay</b>'
                f'<span class="pa-date-tag"><i class="bi bi-calendar3"></i> {today:%d/%m/%Y}</span></div>'
                '<div class="pa-scroll"><table class="pa-table pa-today-table"><thead><tr>'
                '<th>Nhân viên</th><th>Giờ vào · CHECK-IN</th><th>Giờ ra · CHECK-OUT</th><th>Ghi nhận hôm nay</th>'
                f'</tr></thead><tbody><tr><td>{identity}</td><td>{time_cell(record.get("check_in"))}</td>'
                f'<td>{time_cell(record.get("check_out"))}</td><td><span class="ws-legend-{tone}">'
                f'{"Đã CHECK-IN / OUT" if checked_in and checked_out else "Đã CHECK-IN" if checked_in else "Chưa CHECK-IN"}</span></td></tr></tbody></table></div>'
                f'<div class="pa-footer"><i class="bi bi-info-circle"></i> {message}</div></div>', unsafe_allow_html=True)
    with st.container(key="pa_toolbar"):
        left, right, refresh = st.columns([1, 1, .8])
        start = left.date_input("Từ ngày", today - timedelta(days=30), format="DD/MM/YYYY")
        end = right.date_input("Đến ngày", today, format="DD/MM/YYYY")
        with refresh:
            st.write("")
            if st.button("Cập nhật", icon=":material/refresh:", width="stretch"):
                st.rerun()
        st.markdown('<div class="ws-legend"><span class="ws-legend-title"><i class="bi bi-info-circle-fill"></i> Chú thích trạng thái:</span>'
                    '<span class="ws-legend-on">Đúng giờ / Đã ghi nhận</span><span class="ws-legend-off">Đi muộn / Về sớm</span>'
                    '<span class="ws-legend-empty">Chưa ghi nhận</span></div>', unsafe_allow_html=True)
    if start > end:
        st.warning("Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.")
        return
    rows = db.list_attendance(start.isoformat(), end.isoformat(), employee_id=owner)
    statuses = {"ON_TIME": ("Đúng giờ", "on"), "LATE": ("Đi muộn", "off"), "EARLY_LEAVE": ("Về sớm", "off"), "ABSENT": ("Vắng", "empty"), "LATE_EARLY": ("Muộn / về sớm", "off")}
    body = []
    for row in rows:
        label, color = statuses.get(row["status"], (row["status"], "empty"))
        day = date.fromisoformat(row["date"]).strftime("%d/%m/%Y")
        work_day = date.fromisoformat(row["date"])
        weekday = "Chủ nhật" if work_day.weekday() == 6 else f"Thứ {work_day.weekday() + 2}"
        today_tag = '<span class="pa-today-tag">Hôm nay</span>' if work_day == today else ''
        body.append(f'<tr class="{"pa-current-day" if work_day == today else ""}"><td><div class="pa-day"><b>{day}</b>{today_tag}<small>{weekday}</small></div></td>'
                    f'<td>{time_cell(row.get("check_in"))}</td><td>{time_cell(row.get("check_out"))}</td><td><span class="ws-legend-{color}">{escape(label)}</span></td></tr>')
    if not body:
        body.append('<tr><td colspan="4" class="pa-empty">Chưa có dữ liệu chấm công trong khoảng ngày đã chọn.</td></tr>')
    st.markdown('<div class="pa-history"><div class="pa-history-heading"><b>Lịch sử chấm công</b><small>Chỉ hiển thị dữ liệu của bạn</small></div>'
                '<div class="pa-scroll"><table class="pa-table"><thead><tr><th>Ngày</th><th>CHECK-IN</th><th>CHECK-OUT</th><th>Trạng thái</th></tr></thead><tbody>'
                + ''.join(body) + '</tbody></table></div>'
                + f'<div class="pa-footer">Hiển thị {len(rows)} bản ghi · {start:%d/%m/%Y} – {end:%d/%m/%Y}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ws-tip"><i class="bi bi-info-circle"></i><span><b>Lưu ý chấm công</b><small>Dữ liệu được ghi nhận từ camera theo lịch làm việc. Chưa CHECK-IN không đồng nghĩa vắng mặt.</small></span></div>', unsafe_allow_html=True)
