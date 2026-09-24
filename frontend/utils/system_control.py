"""Safe lifecycle control for the tracking backend and Be Xinh voice chat."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


class SystemControlError(RuntimeError):
    pass


@dataclass(frozen=True)
class ServiceStatus:
    running: bool
    healthy: bool
    pid: int | None
    detail: str


@dataclass(frozen=True)
class VoiceVolumeSettings:
    percent: int = 85


class SystemController:
    """Control services while leaving the Streamlit dashboard alive."""

    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace or Path(__file__).resolve().parents[2]
        self.backend_dir = self.workspace / "backend"
        self.python = self.backend_dir / ".venv" / "Scripts" / "python.exe"
        self.runtime_dir = self.workspace / ".runtime"
        self.state_path = self.runtime_dir / "processes.json"
        self.volume_path = self.runtime_dir / "be-xinh-volume.json"
        self.log_dir = self.workspace / "logs"
        self.backend_url = "http://127.0.0.1:8765/status.json"

    def voice_volume(self) -> VoiceVolumeSettings:
        try:
            payload = json.loads(self.volume_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return VoiceVolumeSettings()
        # Read the old normal_percent key once so existing installations keep
        # their chosen volume after removing the lunch-time profile.
        value = payload.get("percent", payload.get("normal_percent"))
        return VoiceVolumeSettings(percent=self._percent(value, 85))

    def save_voice_volume(self, settings: VoiceVolumeSettings) -> VoiceVolumeSettings:
        normalized = VoiceVolumeSettings(
            percent=self._percent(settings.percent, 85),
        )
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.volume_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({
                "percent": normalized.percent,
                "updated_at": datetime.now().astimezone().isoformat(),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.volume_path)
        return normalized

    @staticmethod
    def _percent(value: object, default: int) -> int:
        try:
            return max(10, min(100, int(value)))
        except (TypeError, ValueError):
            return default

    def backend_status(self) -> ServiceStatus:
        pid = self._pid("backend")
        running = bool(pid and self._is_named_process(pid, "run_workstate.py"))
        healthy = self._http_ok(self.backend_url) if running else False
        detail = "Camera AI sẵn sàng" if healthy else (
            "Tiến trình đang khởi động/mất camera" if running else "Đã tắt")
        return ServiceStatus(running, healthy, pid if running else None, detail)

    def chat_status(self) -> ServiceStatus:
        pid = self._pid("chat")
        running = bool(pid and self._is_named_process(pid, "be_xinh_assistant.py"))
        return ServiceStatus(
            running, running, pid if running else None,
            "Đang nghe câu gọi" if running else "Đã tắt",
        )

    def start_backend(self) -> ServiceStatus:
        current = self.backend_status()
        if current.running:
            return current
        self._validate_runtime()
        process = self._spawn(
            "backend",
            [
                "-X", "utf8", "-u", "scripts/run_workstate.py",
                "--stream-host", "0.0.0.0", "--stream-port", "8765",
                "--no-greet", "--no-halinh", "--no-supervisor",
            ],
            "backend.out.log", "backend.err.log",
        )
        self._write_pid("backend", process.pid)
        return ServiceStatus(True, False, process.pid, "Đang khởi động")

    def stop_backend(self) -> ServiceStatus:
        self._stop_verified("backend", "run_workstate.py")
        return ServiceStatus(False, False, None, "Đã tắt")

    def restart_backend(self) -> ServiceStatus:
        self.stop_backend()
        time.sleep(0.4)
        return self.start_backend()

    def start_chat(self) -> ServiceStatus:
        current = self.chat_status()
        if current.running:
            return current
        self._validate_runtime()
        env_path = self.backend_dir / ".env"
        if "GEMINI_API_KEY=" not in env_path.read_text(encoding="utf-8-sig"):
            raise SystemControlError("Chưa cấu hình GEMINI_API_KEY trong backend/.env")

        # Voice chat exclusively owns camera talkback. Stop the old auto-greet
        # bridge if it is still present.
        audio_pid = self._pid("audio")
        if audio_pid and self._is_named_process(audio_pid, "run_be_xinh_bridge.py"):
            self._kill_tree(audio_pid)
        self._write_pid("audio", 0)

        process = self._spawn(
            "chat",
            [
                "-X", "utf8", "-u", "scripts/be_xinh_assistant.py",
                "--model", "gemini-3.5-flash-lite",
            ],
            "be-xinh-chat.out.log", "be-xinh-chat.err.log",
        )
        self._write_pid("chat", process.pid)
        return ServiceStatus(True, True, process.pid, "Đang khởi động")

    def stop_chat(self) -> ServiceStatus:
        self._stop_verified("chat", "be_xinh_assistant.py")
        return ServiceStatus(False, False, None, "Đã tắt")

    def restart_chat(self) -> ServiceStatus:
        self.stop_chat()
        time.sleep(0.4)
        return self.start_chat()

    def _validate_runtime(self) -> None:
        if not self.python.is_file():
            raise SystemControlError(f"Không tìm thấy Python backend: {self.python}")
        if not (self.backend_dir / ".env").is_file():
            raise SystemControlError("Thiếu backend/.env")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _spawn(self, _name: str, args: list[str], out_name: str, err_name: str):
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        stdout = (self.log_dir / out_name).open("ab", buffering=0)
        stderr = (self.log_dir / err_name).open("ab", buffering=0)
        try:
            process = subprocess.Popen(  # noqa: S603
                [str(self.python), *args], cwd=self.backend_dir,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=flags,
            )
        except OSError as error:
            stdout.close()
            stderr.close()
            raise SystemControlError(f"Không thể khởi động dịch vụ: {error}") from error
        stdout.close()
        stderr.close()
        return process

    def _stop_verified(self, name: str, script_name: str) -> None:
        pid = self._pid(name)
        if pid and self._is_named_process(pid, script_name):
            self._kill_tree(pid)
        self._write_pid(name, 0)

    def _kill_tree(self, pid: int) -> None:
        if os.name == "nt":
            result = subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True, text=True, timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW, check=False,
            )
            if result.returncode and self._process_exists(pid):
                raise SystemControlError((result.stderr or result.stdout).strip())
            return
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _pid(self, name: str) -> int | None:
        try:
            value = int(self._read_state().get(name) or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _write_pid(self, name: str, pid: int) -> None:
        state = self._read_state()
        state[name] = int(pid)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _read_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _is_named_process(self, pid: int, script_name: str) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            command = (
                "$p=Get-CimInstance Win32_Process -Filter \"ProcessId="
                f"{pid}\" -ErrorAction SilentlyContinue;"
                "if ($null -ne $p) {$p.CommandLine}"
            )
            try:
                result = subprocess.run(  # noqa: S603
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            return result.returncode == 0 and script_name in result.stdout
        try:
            return script_name in (Path("/proc") / str(pid) / "cmdline").read_text(errors="ignore")
        except OSError:
            return False

    def _process_exists(self, pid: int) -> bool:
        if os.name == "nt":
            return self._is_named_process(pid, ".py")
        return (Path("/proc") / str(pid)).exists()

    @staticmethod
    def _http_ok(url: str) -> bool:
        try:
            with urlopen(Request(url, headers={"User-Agent": "camera-control/1"}), timeout=1.0) as response:  # noqa: S310
                return 200 <= int(response.status) < 300
        except OSError:
            return False
