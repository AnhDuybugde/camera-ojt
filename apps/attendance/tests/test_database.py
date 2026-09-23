from pathlib import Path

import numpy as np

from database.db import Database
from face.embedding import embedding_to_blob


def employee() -> dict[str, str]:
    return {"employee_id": "NV001", "full_name": "Nguyen Van A", "department": "AI", "position": "Engineer"}


def test_employee_crud_and_embedding(tmp_path: Path) -> None:
    db = Database(tmp_path / "test.db")
    db.add_employee(employee())
    assert db.get_employee("NV001")["full_name"] == "Nguyen Van A"
    db.update_employee("NV001", {**employee(), "full_name": "Nguyen Van B"}, actor_role="ADMIN")
    assert db.get_employee("NV001")["full_name"] == "Nguyen Van B"
    blob, size = embedding_to_blob(np.ones(4))
    db.save_embedding("NV001", blob, size)
    assert db.get_employee("NV001")["has_face"] == 1
    db.delete_employee("NV001", actor_role="ADMIN")
    assert db.get_employee("NV001") is None


def test_schema_is_created_automatically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "attendance.db"
    db = Database(path)
    assert path.exists()
    assert db.list_employees() == []
