from pathlib import Path

from ui.audit_log import ACTION_LABELS, ROLE_LABELS, _summary, _timestamp


def test_audit_log_labels_are_vietnamese() -> None:
    assert ROLE_LABELS["EMPLOYEE"] == "Nhân viên"
    assert ACTION_LABELS["SCHEDULE_UPDATE"] == "Cập nhật lịch"
    assert ACTION_LABELS["PASSWORD_RESET"] == "Đặt lại mật khẩu"
    assert ACTION_LABELS["UPDATE_ATTENDANCE"] == "Sửa chấm công"


def test_audit_log_summary_localizes_fields_values_and_dates() -> None:
    result = _summary(
        '{"date":"2026-09-14","check_in":"2026-09-14T10:20:00","status":"LATE"}'
    )

    assert "Ngày: 14/09/2026" in result
    assert "Giờ vào: 14/09 10:20" in result
    assert "Trạng thái: Đi muộn" in result


def test_audit_log_timestamp_is_compact() -> None:
    assert _timestamp("2026-09-23T08:45:12") == "23/09/2026 08:45"


def test_table_cells_and_badges_cannot_overflow() -> None:
    css = Path("assets/style.css").read_text(encoding="utf-8")

    assert ".ds-table td {" in css
    assert "overflow: hidden;" in css
    assert "max-width: 100%; overflow: hidden; text-overflow: ellipsis;" in css
