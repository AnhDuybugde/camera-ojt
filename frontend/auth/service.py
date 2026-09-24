"""Role + employee authentication backed by bcrypt hashes."""
from __future__ import annotations

import bcrypt
import os
import secrets
import time

from auth.permissions import ADMIN, EMPLOYEE
from database.db import Database
from utils.helpers import iso_now


DEVELOPMENT_DEFAULT_PASSWORDS = {EMPLOYEE: "123", ADMIN: "456"}
VALID_ROLES = frozenset(DEVELOPMENT_DEFAULT_PASSWORDS)
WEAK_PASSWORDS = frozenset({"123", "456", "password", "admin", "admin123"})


class AuthConfigurationError(RuntimeError):
    """Raised when production authentication cannot fail closed."""


class AuthService:
    def __init__(self, db: Database, *, production: bool | None = None) -> None:
        self.db = db
        self.production = (
            os.getenv("APP_ENV", "development").strip().lower() == "production"
            if production is None
            else bool(production)
        )
        self.initialize_defaults()
        if not self.db.is_postgres:
            self.db.execute("""CREATE TABLE IF NOT EXISTS employee_accounts (
                employee_id TEXT PRIMARY KEY REFERENCES employees(employee_id) ON DELETE CASCADE,
                password_hash TEXT NOT NULL, must_change INTEGER NOT NULL DEFAULT 1,
                failures INTEGER NOT NULL DEFAULT 0, locked_until REAL NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1)""")
            self.db.execute("""CREATE TABLE IF NOT EXISTS login_limits (
                account TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0,
                locked_until REAL NOT NULL DEFAULT 0)""")
        self.ensure_accounts()

    def ensure_accounts(self) -> None:
        rows = self.db.fetch_all("SELECT employee_id FROM employees WHERE employee_id NOT IN (SELECT employee_id FROM employee_accounts)")
        if rows:
            hashed = self._hash(self._bootstrap_password(EMPLOYEE))
            with self.db.transaction() as conn:
                for row in rows:
                    conn.execute("INSERT INTO employee_accounts(employee_id,password_hash) VALUES (?,?) ON CONFLICT (employee_id) DO NOTHING", (row["employee_id"], hashed))

    def account(self, employee_id: str) -> dict | None:
        return self.db.fetch_one("SELECT * FROM employee_accounts WHERE employee_id=?", (employee_id,))

    def login(self, role: str, password: str, employee_id: str = "") -> dict | None:
        self.ensure_accounts()
        key = "ADMIN" if role == ADMIN else "EMPLOYEE:" + employee_id.strip()
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO login_limits(account) VALUES (?) ON CONFLICT (account) DO NOTHING", (key,))
            limit = conn.execute("SELECT * FROM login_limits WHERE account=?", (key,)).fetchone()
            if limit["locked_until"] > time.time():
                raise ValueError("Nhập sai quá nhiều lần. Hãy thử lại sau 5 phút.")
            account = self.account(employee_id.strip()) if role == EMPLOYEE else None
            valid = self._verify(password, account["password_hash"]) if account else (role == ADMIN and self.authenticate(ADMIN, password))
            if not valid:
                failures = limit["failures"] + 1 if limit["locked_until"] == 0 else 1
                conn.execute("UPDATE login_limits SET failures=?, locked_until=? WHERE account=?", (failures, time.time() + 300 if failures >= 5 else 0, key))
                return None
            conn.execute("UPDATE login_limits SET failures=0,locked_until=0 WHERE account=?", (key,))
            return account if role == EMPLOYEE else {"role": ADMIN}

    def change_employee_password(self, employee_id: str, current: str, new: str, confirmation: str) -> None:
        row = self.account(employee_id)
        if not row or not self._verify(current, row["password_hash"]):
            raise ValueError("Mật khẩu hiện tại không chính xác.")
        self._validate_new_password(new)
        if new != confirmation:
            raise ValueError("Xác nhận mật khẩu không khớp.")
        self.db.execute("UPDATE employee_accounts SET password_hash=?,must_change=0,version=version+1 WHERE employee_id=?", (self._hash(new), employee_id))

    def reset_employee_password(self, employee_id: str, *, actor_role: str) -> str:
        from auth.permissions import require_permission
        require_permission(actor_role, "account.reset")
        temporary_password = secrets.token_urlsafe(12)
        with self.db.transaction() as conn:
            conn.execute("UPDATE employee_accounts SET password_hash=?,must_change=1,version=version+1 WHERE employee_id=?", (self._hash(temporary_password), employee_id))
            conn.execute("DELETE FROM login_limits WHERE account=?", ("EMPLOYEE:" + employee_id,))
            conn.execute("INSERT INTO audit_logs(timestamp,role,action,employee_id) VALUES (?,?,?,?)", (iso_now(), actor_role, "PASSWORD_RESET", employee_id))
        return temporary_password

    @staticmethod
    def _role(role: str) -> str:
        normalized = str(role).strip().upper()
        if normalized not in VALID_ROLES:
            raise ValueError("Vai trò không hợp lệ.")
        return normalized

    @staticmethod
    def _hash(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")

    @staticmethod
    def _verify(password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
        except (ValueError, TypeError):
            return False

    def initialize_defaults(self) -> None:
        existing = {
            row["role"] for row in self.db.fetch_all("SELECT role FROM auth_settings")
        }
        missing = [role for role in VALID_ROLES if role not in existing]
        if not missing:
            return
        now = iso_now()
        with self.db.transaction() as conn:
            for role in missing:
                conn.execute(
                    """INSERT INTO auth_settings(role, password_hash, updated_at)
                       VALUES (?, ?, ?) ON CONFLICT (role) DO NOTHING""",
                    (role, self._hash(self._bootstrap_password(role)), now),
                )

    def _bootstrap_password(self, role: str) -> str:
        if not self.production:
            return DEVELOPMENT_DEFAULT_PASSWORDS[role]
        env_name = (
            "ADMIN_BOOTSTRAP_PASSWORD"
            if role == ADMIN
            else "EMPLOYEE_BOOTSTRAP_PASSWORD"
        )
        password = os.getenv(env_name, "").strip()
        try:
            self._validate_new_password(password)
        except ValueError as error:
            raise AuthConfigurationError(
                f"{env_name} must be configured with a strong production password: {error}"
            ) from error
        return password

    def _validate_new_password(self, password: str) -> None:
        minimum = 12 if self.production else 8
        if len(password) < minimum or len(password.encode("utf-8")) > 72:
            raise ValueError(
                f"Mật khẩu cần ít nhất {minimum} ký tự và tối đa 72 byte."
            )
        if password.lower() in WEAK_PASSWORDS:
            raise ValueError("Mật khẩu quá phổ biến.")
        if self.production:
            groups = (
                any(char.islower() for char in password),
                any(char.isupper() for char in password),
                any(char.isdigit() for char in password),
                any(not char.isalnum() for char in password),
            )
            if sum(groups) < 3:
                raise ValueError(
                    "Mật khẩu production phải có ít nhất 3 nhóm: chữ thường, "
                    "chữ hoa, số, ký tự đặc biệt."
                )

    def authenticate(self, role: str, password: str) -> bool:
        normalized = self._role(role)
        row = self.db.fetch_one(
            "SELECT password_hash FROM auth_settings WHERE role=?", (normalized,)
        )
        return bool(row and self._verify(password, row["password_hash"]))

    def change_password(
        self, role: str, current_password: str, new_password: str, confirmation: str
    ) -> None:
        normalized = self._role(role)
        if not self.authenticate(normalized, current_password):
            raise ValueError("Mật khẩu hiện tại không chính xác.")
        self._validate_new_password(new_password)
        if new_password != confirmation:
            raise ValueError("Xác nhận mật khẩu mới không khớp.")
        self.db.execute(
            "UPDATE auth_settings SET password_hash=?, updated_at=? WHERE role=?",
            (self._hash(new_password), iso_now(), normalized),
        )
