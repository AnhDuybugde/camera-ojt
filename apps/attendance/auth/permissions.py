"""Central role/permission policy."""
from __future__ import annotations


ADMIN = "ADMIN"
EMPLOYEE = "EMPLOYEE"
SYSTEM = "SYSTEM"


class PermissionDenied(PermissionError):
    pass


_EMPLOYEE_PERMISSIONS = frozenset({
    "dashboard.view",
    "camera.view",
    "face.register",
    "schedule.view",
    "schedule.update",
})


def has_permission(role: str, permission: str) -> bool:
    role = str(role).upper()
    return role in {ADMIN, SYSTEM} or (
        role == EMPLOYEE and permission in _EMPLOYEE_PERMISSIONS
    )


def require_permission(role: str, permission: str) -> None:
    if not has_permission(role, permission):
        raise PermissionDenied("Bạn không có quyền thực hiện thao tác này.")


def require_owner(role: str, target: str, employee_id: str | None) -> None:
    if role not in {ADMIN, SYSTEM} and (role != EMPLOYEE or not employee_id or target != employee_id):
        raise PermissionDenied("Bạn chỉ được thao tác dữ liệu của chính mình.")
