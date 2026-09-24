from __future__ import annotations

from types import SimpleNamespace

from utils import be_xinh_control
from utils.be_xinh_control import BeXinhControlError, BeXinhController


def _workspace(tmp_path):
    workspace = tmp_path / "team-integration"
    python = workspace / "backend" / ".venv" / "Scripts" / "python.exe"
    script = workspace / "backend" / "scripts" / "run_be_xinh_bridge.py"
    python.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    python.touch()
    script.touch()
    return workspace


def test_start_records_audio_pid_and_preference(tmp_path, monkeypatch):
    controller = BeXinhController(_workspace(tmp_path))
    monkeypatch.setattr(controller, "_is_bridge_process", lambda pid: False)

    recorded = {}

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    def fake_popen(arguments, **options):
        recorded["arguments"] = arguments
        recorded["options"] = options
        return FakeProcess()

    monkeypatch.setattr(be_xinh_control.subprocess, "Popen", fake_popen)
    status = controller.start()

    assert status.running is True
    assert status.pid == 4321
    assert controller._read_json(controller.process_state)["audio"] == 4321
    assert controller._read_json(controller.preference_state)["enabled"] is True
    assert "run_be_xinh_bridge.py" in " ".join(recorded["arguments"])


def test_stop_kills_only_verified_bridge_tree(tmp_path, monkeypatch):
    controller = BeXinhController(_workspace(tmp_path))
    controller._write_audio_pid(9876)
    running = {"value": True}
    monkeypatch.setattr(controller, "_is_bridge_process", lambda pid: running["value"])

    def fake_run(arguments, **options):
        assert arguments == ["taskkill", "/PID", "9876", "/T", "/F"]
        running["value"] = False
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(be_xinh_control.subprocess, "run", fake_run)
    status = controller.stop()

    assert status.running is False
    assert controller._read_json(controller.process_state)["audio"] == 0
    assert controller._read_json(controller.preference_state)["enabled"] is False


def test_stale_pid_is_reported_as_stopped(tmp_path, monkeypatch):
    controller = BeXinhController(_workspace(tmp_path))
    controller._write_audio_pid(1234)
    monkeypatch.setattr(controller, "_is_bridge_process", lambda pid: False)

    assert controller.status().running is False


def test_start_refuses_greeting_while_voice_chat_is_running(tmp_path, monkeypatch):
    controller = BeXinhController(_workspace(tmp_path))
    controller._write_json(controller.process_state, {"audio": 0, "chat": 2468})
    monkeypatch.setattr(
        controller,
        "_is_named_process",
        lambda pid, script: pid == 2468 and script == "be_xinh_assistant.py",
    )

    try:
        controller.start()
    except BeXinhControlError as error:
        assert "Voice chat" in str(error)
    else:
        raise AssertionError("Greeting bridge must not start during voice chat")
    assert controller._read_json(controller.preference_state)["enabled"] is False
