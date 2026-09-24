"""Start and stop the Be Xinh audio bridge without touching camera services."""
from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class BeXinhControlError(RuntimeError):
    """Raised when the audio bridge cannot be safely controlled."""


@dataclass(frozen=True)
class BeXinhStatus:
    running: bool
    pid: int | None


class BeXinhController:
    """Own the lifecycle and persisted preference of the Be Xinh bridge."""

    def __init__(self, workspace: Path | None = None, status_url: str = "http://127.0.0.1:8765/status.json") -> None:
        self.workspace = workspace or Path(__file__).resolve().parents[2]
        self.backend_dir = self.workspace / "backend"
        self.python = self.backend_dir / ".venv" / "Scripts" / "python.exe"
        self.bridge_script = self.backend_dir / "scripts" / "run_be_xinh_bridge.py"
        self.runtime_dir = self.workspace / ".runtime"
        self.process_state = self.runtime_dir / "processes.json"
        self.preference_state = self.runtime_dir / "be-xinh.json"
        self.log_dir = self.workspace / "logs"
        self.status_url = status_url

    def status(self) -> BeXinhStatus:
        pid = self._audio_pid()
        if pid and self._is_bridge_process(pid):
            return BeXinhStatus(running=True, pid=pid)
        return BeXinhStatus(running=False, pid=None)

    def start(self) -> BeXinhStatus:
        if self._chat_running():
            self._write_preference(False)
            raise BeXinhControlError(
                "Voice chat Bé Xinh đang chạy; đã giữ chào tự động ở trạng thái tắt để tránh nói chồng."
            )
        current = self.status()
        if current.running:
            self._write_preference(True)
            return current
        if not self.python.is_file():
            raise BeXinhControlError(f"Không tìm thấy Python backend: {self.python}")
        if not self.bridge_script.is_file():
            raise BeXinhControlError(f"Không tìm thấy bridge Bé Xinh: {self.bridge_script}")

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        with (self.log_dir / "be-xinh.out.log").open("ab", buffering=0) as stdout_file, (
            self.log_dir / "be-xinh.err.log"
        ).open("ab", buffering=0) as stderr_file:
            try:
                process = subprocess.Popen(  # noqa: S603
                    [
                        str(self.python),
                        "-X",
                        "utf8",
                        "-u",
                        str(self.bridge_script),
                        "--status-url",
                        self.status_url,
                    ],
                    cwd=self.backend_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    creationflags=creationflags,
                )
            except OSError as error:
                raise BeXinhControlError(f"Không thể bật Bé Xinh: {error}") from error

        if process.poll() is not None:
            raise BeXinhControlError("Bé Xinh dừng ngay khi khởi động. Hãy xem logs/be-xinh.err.log.")
        self._write_audio_pid(process.pid)
        self._write_preference(True)
        return BeXinhStatus(running=True, pid=process.pid)

    def stop(self) -> BeXinhStatus:
        current = self.status()
        if current.pid is not None:
            self._stop_process_tree(current.pid)
        self._write_audio_pid(0)
        self._write_preference(False)
        return BeXinhStatus(running=False, pid=None)

    def _stop_process_tree(self, pid: int) -> None:
        if not self._is_bridge_process(pid):
            raise BeXinhControlError("PID đã thay đổi; từ chối dừng tiến trình không phải Bé Xinh.")
        if os.name == "nt":
            result = subprocess.run(  # noqa: S603
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
            if result.returncode and self._is_bridge_process(pid):
                detail = (result.stderr or result.stdout).strip()
                raise BeXinhControlError(f"Không thể tắt Bé Xinh: {detail}")
            return
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError) as error:
            if self._is_bridge_process(pid):
                raise BeXinhControlError(f"Không thể tắt Bé Xinh: {error}") from error

    def _audio_pid(self) -> int | None:
        payload = self._read_json(self.process_state)
        try:
            pid = int(payload.get("audio") or 0)
        except (TypeError, ValueError):
            return None
        return pid if pid > 0 else None

    def _chat_running(self) -> bool:
        payload = self._read_json(self.process_state)
        try:
            pid = int(payload.get("chat") or 0)
        except (TypeError, ValueError):
            return False
        return pid > 0 and self._is_named_process(pid, "be_xinh_assistant.py")

    def _is_bridge_process(self, pid: int) -> bool:
        return self._is_named_process(pid, "run_be_xinh_bridge.py")

    def _is_named_process(self, pid: int, script_name: str) -> bool:
        if pid <= 0:
            return False
        if os.name == "nt":
            command = (
                "$p=Get-CimInstance Win32_Process -Filter \"ProcessId="
                f"{int(pid)}\" -ErrorAction SilentlyContinue;"
                "if ($null -ne $p) {$p.CommandLine}"
            )
            try:
                result = subprocess.run(  # noqa: S603
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            return result.returncode == 0 and script_name in result.stdout
        try:
            command_line = (Path("/proc") / str(pid) / "cmdline").read_text(errors="ignore")
        except OSError:
            return False
        return script_name in command_line

    def _write_audio_pid(self, pid: int) -> None:
        payload = self._read_json(self.process_state)
        payload["audio"] = int(pid)
        self._write_json(self.process_state, payload)

    def _write_preference(self, enabled: bool) -> None:
        self._write_json(
            self.preference_state,
            {"enabled": enabled, "updated_at": datetime.now().astimezone().isoformat()},
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
