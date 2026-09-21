from pathlib import Path

import pytest

from auth.permissions import ADMIN, EMPLOYEE, PermissionDenied, require_permission
from auth.service import AuthService
from database.db import Database


def test_default_passwords_are_hashed_and_authenticate(tmp_path: Path) -> None:
    db = Database(tmp_path / "auth.db")
    auth = AuthService(db)
    assert auth.authenticate(EMPLOYEE, "123")
    assert auth.authenticate(ADMIN, "456")
    assert not auth.authenticate(EMPLOYEE, "456")
    assert not auth.authenticate(ADMIN, "123")
    assert not auth.authenticate(ADMIN, "wrong")
    for row in db.fetch_all("SELECT password_hash FROM auth_settings"):
        assert row["password_hash"] not in {"123", "456"}
        assert row["password_hash"].startswith("$2")


def test_password_change_persists_and_defaults_are_not_reset(tmp_path: Path) -> None:
    path = tmp_path / "auth.db"
    auth = AuthService(Database(path))
    auth.change_password(EMPLOYEE, "123", "new-employee", "new-employee")
    restarted = AuthService(Database(path))
    assert restarted.authenticate(EMPLOYEE, "new-employee")
    assert not restarted.authenticate(EMPLOYEE, "123")
    assert restarted.authenticate(ADMIN, "456")


def test_password_change_validation(tmp_path: Path) -> None:
    auth = AuthService(Database(tmp_path / "auth.db"))
    with pytest.raises(ValueError):
        auth.change_password(ADMIN, "wrong", "new", "new")
    with pytest.raises(ValueError):
        auth.change_password(ADMIN, "456", "new", "different")


def test_admin_password_change_invalidates_default(tmp_path: Path) -> None:
    auth = AuthService(Database(tmp_path / "admin-auth.db"))
    auth.change_password(ADMIN, "456", "new-admin", "new-admin")
    assert auth.authenticate(ADMIN, "new-admin")
    assert not auth.authenticate(ADMIN, "456")


def test_employee_permissions_block_sensitive_operations(tmp_path: Path) -> None:
    db = Database(tmp_path / "permissions.db")
    db.add_employee({"employee_id": "NV001", "full_name": "Nguyen Van A"})
    with pytest.raises(PermissionDenied):
        require_permission(EMPLOYEE, "employee.add")
    require_permission(EMPLOYEE, "face.register")
    require_permission(EMPLOYEE, "schedule.update")
    with pytest.raises(PermissionDenied):
        db.update_employee(
            "NV001", {"full_name": "Changed"}, actor_role=EMPLOYEE
        )
    with pytest.raises(PermissionDenied):
        db.delete_employee("NV001", actor_role=EMPLOYEE)
    for permission in ("settings.view", "sync.manage", "audit.view"):
        with pytest.raises(PermissionDenied):
            require_permission(EMPLOYEE, permission)
    db.update_employee("NV001", {"full_name": "Admin changed"}, actor_role=ADMIN)
    assert db.get_employee("NV001")["full_name"] == "Admin changed"


def test_individual_accounts_reset_and_lockout(tmp_path):
    db = Database(tmp_path / "individual.db")
    db.add_employee({"employee_id": "NV001", "full_name": "A"})
    auth = AuthService(db)
    assert auth.login(EMPLOYEE, "123", "NV001")["must_change"] == 1
    assert auth.login(EMPLOYEE, "123", "") is None
    auth.change_employee_password("NV001", "123", "password123", "password123")
    assert auth.login(EMPLOYEE, "password123", "NV001")["must_change"] == 0
    with pytest.raises(PermissionDenied):
        auth.reset_employee_password("NV001", actor_role=EMPLOYEE)
    auth.reset_employee_password("NV001", actor_role=ADMIN)
    assert auth.login(EMPLOYEE, "123", "NV001")["must_change"] == 1
    for _ in range(5):
        assert auth.login(EMPLOYEE, "wrong", "NV001") is None
    with pytest.raises(ValueError):
        auth.login(EMPLOYEE, "123", "NV001")


def test_owner_enforced_at_storage(tmp_path):
    db = Database(tmp_path / "owner.db")
    for emp in ("NV001", "NV002"):
        db.add_employee({"employee_id": emp, "full_name": emp})
    entries = [("NV002", "2026-09-16", "MORNING", "ON")]
    with pytest.raises(PermissionDenied):
        db.save_work_schedules(entries, actor_role=EMPLOYEE, actor_employee_id="NV001")
    with pytest.raises(PermissionDenied):
        db.save_embedding("NV002", b"test", 1, actor_role=EMPLOYEE, actor_employee_id="NV001")
    db.save_embedding("NV001", b"test", 1, actor_role=EMPLOYEE, actor_employee_id="NV001")
    with pytest.raises(PermissionError):
        db.save_embedding("NV001", b"replace", 1, actor_role=EMPLOYEE, actor_employee_id="NV001")
    db.save_embedding("NV001", b"replace", 1, actor_role=ADMIN)
