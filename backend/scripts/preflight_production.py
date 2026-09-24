"""Production preflight for the camera tracking stack.

Runs without opening cameras or loading heavy AI models. It checks configuration,
secrets, storage paths, optional GPU providers, and known release blockers.

Usage:
    python scripts/preflight_production.py
    python scripts/preflight_production.py --strict
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from camera_tracking.config import load_config

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = ROOT.parent / "frontend"


class Report:
    def __init__(self) -> None:
        self.ok = 0
        self.warn = 0
        self.block = 0

    def pass_(self, message: str) -> None:
        self.ok += 1
        print(f"[PASS] {message}")

    def warning(self, message: str) -> None:
        self.warn += 1
        print(f"[WARN] {message}")

    def blocker(self, message: str) -> None:
        self.block += 1
        print(f"[BLOCK] {message}")


def _has_env(name: str) -> bool:
    value = os.getenv(name, "").strip()
    return bool(value and not value.lower().startswith("your_"))


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip() == "1"


def _writable_parent(path: Path) -> bool:
    parent = path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        probe = parent / ".preflight-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _strong_password(password: str) -> bool:
    if len(password) < 12 or len(password.encode("utf-8")) > 72:
        return False
    groups = (
        any(char.islower() for char in password),
        any(char.isupper() for char in password),
        any(char.isdigit() for char in password),
        any(not char.isalnum() for char in password),
    )
    return sum(groups) >= 3 and password.lower() not in {
        "123", "456", "password", "admin", "admin123"
    }


def _matches_known_weak_password(password_hash: str, bcrypt_module) -> bool:
    try:
        encoded_hash = str(password_hash).encode("ascii")
        return any(
            bcrypt_module.checkpw(candidate.encode("utf-8"), encoded_hash)
            for candidate in ("123", "456", "password", "admin", "admin123")
        )
    except (TypeError, ValueError, UnicodeEncodeError):
        return True


def _check_frontend_auth(report: Report, *, strict: bool) -> None:
    """Detect unsafe bootstrap/runtime auth settings without printing secrets."""
    env_path = FRONTEND_ROOT / ".env"
    values = dotenv_values(env_path) if env_path.exists() else {}
    app_env = str(values.get("APP_ENV") or os.getenv("APP_ENV", "development"))
    if app_env.strip().lower() == "production":
        report.pass_("Frontend APP_ENV=production")
    elif strict:
        report.blocker("Frontend APP_ENV must be production for a strict release")
    else:
        report.warning("Frontend is not in production mode")

    database_url = str(values.get("DATABASE_URL") or "sqlite:///data/database.db")
    if not database_url.startswith("sqlite:///"):
        report.warning("Frontend auth DB is remote; verify password rotation externally")
        return
    raw_path = database_url.removeprefix("sqlite:///")
    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = FRONTEND_ROOT / db_path
    if not db_path.exists():
        bootstrap = [
            str(values.get(name) or os.getenv(name, ""))
            for name in ("ADMIN_BOOTSTRAP_PASSWORD", "EMPLOYEE_BOOTSTRAP_PASSWORD")
        ]
        if all(_strong_password(password) for password in bootstrap):
            report.pass_("Strong frontend bootstrap passwords are configured")
        elif strict:
            report.blocker(
                "Frontend auth DB is absent and strong bootstrap passwords are missing"
            )
        else:
            report.warning("Frontend auth DB is absent; configure strong bootstrap passwords")
        return

    try:
        import bcrypt

        with sqlite3.connect(db_path) as connection:
            hashes = [
                row[0]
                for table in ("auth_settings", "employee_accounts")
                for row in connection.execute(f"SELECT password_hash FROM {table}")
            ]
        weak_accounts = sum(
            1
            for password_hash in hashes
            if _matches_known_weak_password(password_hash, bcrypt)
        )
    except (ImportError, sqlite3.Error, TypeError, ValueError) as error:
        report.warning(f"Could not audit frontend password hashes: {type(error).__name__}")
        return
    if weak_accounts and strict:
        report.blocker(f"Frontend contains {weak_accounts} account(s) with weak passwords")
    elif weak_accounts:
        report.warning(f"Frontend contains {weak_accounts} account(s) with weak passwords")
    else:
        report.pass_(f"No known weak password found across {len(hashes)} frontend accounts")


def run(config_path: Path, *, strict: bool) -> int:
    report = Report()
    load_dotenv(ROOT / ".env")

    if sys.version_info >= (3, 10):
        report.pass_(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        report.blocker("Python >= 3.10 is required")

    try:
        config = load_config(config_path)
        report.pass_(f"Config valid: {config_path}")
    except Exception as error:  # noqa: BLE001 - preflight should report all failures
        report.blocker(f"Config invalid: {error}")
        config = None

    if (ROOT / ".env").exists():
        report.pass_("backend/.env exists")
    else:
        report.blocker("backend/.env missing")

    app_env = os.getenv("APP_ENV", "development").strip().lower()
    if app_env == "production":
        report.pass_("Backend APP_ENV=production")
    elif strict:
        report.blocker("Backend APP_ENV must be production for a strict release")
    else:
        report.warning("Backend is not in production mode")

    _check_frontend_auth(report, strict=strict)

    camera_keys = ("IMOU_IP", "IMOU_USER", "IMOU_PASSWORD")
    missing_camera = [key for key in camera_keys if not _has_env(key)]
    if missing_camera:
        report.warning("Camera credentials incomplete: " + ", ".join(missing_camera))
    else:
        report.pass_("Camera credentials are configured")

    if config is not None:
        if config.store.supabase_enabled:
            missing = [
                key for key in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY")
                if not _has_env(key)
            ]
            if missing:
                report.blocker("Supabase enabled but missing: " + ", ".join(missing))
            else:
                report.pass_("Supabase service credentials are configured")
        else:
            report.warning("Supabase sync is disabled; writes remain local/queued")

        queue_path = ROOT / config.store.queue_db
        if _writable_parent(queue_path):
            report.pass_(f"Queue/output path writable: {queue_path.parent}")
        else:
            report.blocker(f"Queue/output path is not writable: {queue_path.parent}")

        gallery = ROOT / config.face.gallery_dir
        images = []
        if gallery.exists():
            images = [p for p in gallery.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not config.face.enabled:
            report.warning("Face recognition is disabled")
        elif images:
            report.pass_(f"Face gallery present ({len(images)} image files)")
        else:
            report.blocker(f"Face recognition enabled but gallery is empty/missing: {gallery}")

        if config.face.consensus_hits < 2:
            message = (
                "face.consensus_hits=1 is aggressive for biometric attendance; "
                "calibrate on deployment-camera data before release"
            )
            if strict and _env_enabled("ATTENDANCE_AUTOMATION_ENABLED"):
                report.blocker(message)
            else:
                report.warning(message)
        else:
            report.pass_(f"Face temporal consensus: {config.face.consensus_hits} hits")

        if config.face.face_device == "cuda":
            try:
                import onnxruntime as ort
                providers = set(ort.get_available_providers())
                if "CUDAExecutionProvider" in providers:
                    report.pass_("ONNX Runtime CUDA provider available")
                else:
                    report.blocker(
                        "face_device=cuda but CUDAExecutionProvider is unavailable; "
                        "install the face-gpu dependency profile"
                    )
            except ImportError:
                report.blocker(
                    "face_device=cuda but onnxruntime is not installed in this environment"
                )

        if config.identity.reid_device == "cuda" or config.detection.device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    report.pass_(f"PyTorch CUDA available ({torch.cuda.get_device_name(0)})")
                else:
                    report.blocker("Detection/ReID request CUDA but torch.cuda.is_available() is false")
            except ImportError:
                report.warning("PyTorch is not installed in this preflight environment")

    if _env_enabled("CAMERA_ALLOW_REMOTE_STREAM"):
        report.warning("CAMERA_ALLOW_REMOTE_STREAM is enabled; verify reverse proxy/firewall")
    else:
        report.pass_("Remote MJPEG exposure escape hatch is disabled")

    if _env_enabled("CAMERA_ALLOW_REMOTE_CONTROL"):
        report.warning("CAMERA_ALLOW_REMOTE_CONTROL is enabled; pipeline start/stop may be remotely reachable")
    else:
        report.pass_("Remote pipeline-control escape hatch is disabled")

    migration = ROOT / "supabase" / "migrations" / "20260924_production_hardening.sql"
    if migration.exists():
        report.pass_("Production DB migration is present")
    else:
        report.blocker("Production DB migration is missing")

    automatic_attendance = os.getenv(
        "ATTENDANCE_AUTOMATION_ENABLED",
        "false" if app_env == "production" else "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    liveness_provider = os.getenv("BIOMETRIC_LIVENESS_PROVIDER", "").strip()
    if automatic_attendance and strict:
        report.blocker(
            "Automatic biometric attendance cannot pass strict release yet: "
            "the backend liveness/anti-spoof gate is not implemented"
        )
    elif automatic_attendance:
        report.warning(
            "Automatic biometric attendance is enabled before the backend liveness gate "
            f"is implemented (declared provider: {liveness_provider or 'none'})"
        )
    else:
        report.pass_(
            "Automatic biometric attendance is disabled (safe monitor-only mode)"
        )

    print(
        f"\nPreflight summary: {report.ok} pass, {report.warn} warning, "
        f"{report.block} blocker(s)."
    )
    return 2 if report.block else (1 if strict and report.warn else 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera production preflight")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unfinished biometric safety work as a release blocker.",
    )
    args = parser.parse_args()
    raise SystemExit(run(args.config, strict=args.strict))


if __name__ == "__main__":
    main()
