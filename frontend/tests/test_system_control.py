from __future__ import annotations

from types import SimpleNamespace
from utils.system_control import SystemController, VoiceVolumeSettings


def _workspace(tmp_path):
    workspace = tmp_path / "team-integration"
    python = workspace / "backend" / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    (workspace / "backend" / ".env").write_text(
        "GEMINI_API_KEY=test-key-value\n", encoding="utf-8"
    )
    return workspace


def test_backend_status_distinguishes_running_from_healthy(tmp_path, monkeypatch):
    controller = SystemController(_workspace(tmp_path))
    controller._write_pid("backend", 1234)
    monkeypatch.setattr(controller, "_is_named_process", lambda pid, name: True)
    monkeypatch.setattr(controller, "_http_ok", lambda url: False)

    status = controller.backend_status()

    assert status.running is True
    assert status.healthy is False
    assert status.pid == 1234


def test_start_backend_records_pid_and_safe_local_binding(tmp_path, monkeypatch):
    controller = SystemController(_workspace(tmp_path))
    monkeypatch.setattr(controller, "backend_status", lambda: SimpleNamespace(running=False))
    captured = {}

    def fake_spawn(name, args, stdout, stderr):
        captured["args"] = args
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(controller, "_spawn", fake_spawn)
    status = controller.start_backend()

    assert status.pid == 4321
    assert controller._pid("backend") == 4321
    assert "127.0.0.1" in captured["args"]


def test_stop_chat_only_kills_verified_process(tmp_path, monkeypatch):
    controller = SystemController(_workspace(tmp_path))
    controller._write_pid("chat", 9876)
    monkeypatch.setattr(
        controller, "_is_named_process",
        lambda pid, name: pid == 9876 and name == "be_xinh_assistant.py",
    )
    killed = []
    monkeypatch.setattr(controller, "_kill_tree", killed.append)

    status = controller.stop_chat()

    assert killed == [9876]
    assert status.running is False
    assert controller._pid("chat") is None
    assert '"enabled": false' in controller.chat_preference_path.read_text(
        encoding="utf-8"
    ).lower()


def test_voice_volume_persists_single_adjustable_level(tmp_path):
    controller = SystemController(_workspace(tmp_path))
    saved = controller.save_voice_volume(VoiceVolumeSettings(
        percent=90,
    ))

    loaded = controller.voice_volume()
    assert loaded == saved
    assert loaded.percent == 90


def test_voice_volume_migrates_old_normal_level(tmp_path):
    controller = SystemController(_workspace(tmp_path))
    controller.runtime_dir.mkdir(parents=True)
    controller.volume_path.write_text(
        '{"normal_percent":65,"lunch_percent":25,"auto_lunch":true}',
        encoding="utf-8",
    )

    assert controller.voice_volume().percent == 65


def test_start_chat_reads_live_mode_from_dotenv(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    (workspace / "backend" / ".env").write_text(
        "GEMINI_API_KEY=test-key-value\n"
        "BE_XINH_REALTIME_MODE=live\n"
        "GEMINI_LIVE_MODEL=gemini-3.8-live\n",
        encoding="utf-8",
    )
    controller = SystemController(workspace)
    monkeypatch.setattr(
        controller, "chat_status", lambda: SimpleNamespace(running=False)
    )
    captured = {}

    def fake_spawn(name, args, stdout, stderr):
        captured["args"] = args
        return SimpleNamespace(pid=2468)

    monkeypatch.setattr(controller, "_spawn", fake_spawn)
    controller.start_chat()

    assert "scripts/be_xinh_live_assistant.py" in captured["args"]
    assert "gemini-3.8-live" in captured["args"]


def test_start_chat_defaults_to_classic_mode(tmp_path, monkeypatch):
    controller = SystemController(_workspace(tmp_path))
    monkeypatch.setattr(
        controller, "chat_status", lambda: SimpleNamespace(running=False)
    )
    captured = {}

    def fake_spawn(name, args, stdout, stderr):
        captured["args"] = args
        return SimpleNamespace(pid=1357)
    monkeypatch.setattr(controller, "_spawn", fake_spawn)

    controller.start_chat()

    assert "scripts/be_xinh_assistant.py" in captured["args"]
