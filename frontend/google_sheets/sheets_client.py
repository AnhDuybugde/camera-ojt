"""Small Google Sheets API adapter with deterministic upserts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import settings

HEADERS = ["Employee ID", "Employee Name", "Department", "Date", "Check In", "Check Out", "Status"]


class SheetsClient:
    def __init__(self, sheet_id: str | None = None, credentials_path: str | None = None) -> None:
        self.sheet_id = sheet_id if sheet_id is not None else settings.google_sheet_id
        self.credentials_path = credentials_path if credentials_path is not None else settings.google_credentials_path
        self._worksheet: Any = None

    @property
    def configured(self) -> bool:
        return bool(self.sheet_id and self.credentials_path)

    def connect(self) -> Any:
        if self._worksheet is not None:
            return self._worksheet
        if not self.configured:
            raise RuntimeError("Google Sheets is not configured in .env.")
        if not Path(self.credentials_path).expanduser().is_file():
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")
        import gspread

        client = gspread.service_account(filename=str(Path(self.credentials_path).expanduser()))
        spreadsheet = client.open_by_key(self.sheet_id)
        try:
            worksheet = spreadsheet.worksheet(settings.google_worksheet)
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=settings.google_worksheet, rows=1000, cols=10)
        first_row = worksheet.row_values(1)
        if first_row != HEADERS:
            worksheet.update(values=[HEADERS], range_name="A1:G1")
            worksheet.format("A1:G1", {"textFormat": {"bold": True}, "backgroundColor": {"red": .85, "green": .9, "blue": 1}})
            worksheet.freeze(rows=1)
        self._worksheet = worksheet
        return worksheet

    def upsert_attendance(self, record: dict[str, Any]) -> None:
        worksheet = self.connect()
        values = [
            record["employee_id"], record["employee_name"], record.get("department", ""),
            record["date"], _clock(record.get("check_in")), _clock(record.get("check_out")),
            {"WAITING": "Chờ chấm công", "ON_TIME": "Đúng giờ", "LATE": "Đi muộn",
             "ABSENT": "Vắng mặt"}.get(record["status"], record["status"]),
        ]
        # Employee ID + date form the same unique key as the source database.
        rows = worksheet.get_all_values()
        target = next(
            (index for index, row in enumerate(rows[1:], start=2)
             if len(row) >= 4 and row[0] == record["employee_id"] and row[3] == record["date"]),
            None,
        )
        if target:
            worksheet.update(values=[values], range_name=f"A{target}:G{target}")
        else:
            worksheet.append_row(values, value_input_option="USER_ENTERED")


def _clock(value: str | None) -> str:
    if not value:
        return ""
    return value.split("T", 1)[-1][:8]
