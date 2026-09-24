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
import sys
from pathlib import Path

from dotenv import load_dotenv

from camera_tracking.config import load_config

ROOT = Path(__file__).resolve().parents[1]


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
            report.warning(
                "face.consensus_hits=1 is aggressive for biometric attendance; "
                "calibrate on deployment-camera data before release"
            )
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

    # There is no production anti-spoof provider wired into backend attendance yet.
    if strict:
        report.blocker(
            "Biometric liveness/anti-spoof is not wired into the backend attendance gate"
        )
    else:
        report.warning(
            "Liveness/anti-spoof is still a release blocker for automatic biometric attendance"
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
