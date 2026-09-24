from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from halinh_assistant import (
    fast_answer,
    gemini_model_candidates,
    is_gemini_capacity_error,
    is_wake,
    strip_wake_command,
)


def test_be_xinh_name_is_required_for_wake() -> None:
    assert is_wake("Bé Xinh ơi") is True
    assert is_wake("Bé Xinh") is True
    assert is_wake("Bé Xinh ơi, mấy giờ rồi") is True
    assert is_wake("xinh ơi") is False
    assert is_wake("Hà Linh ơi") is False
    assert is_wake("Bé Xinh xin chào bạn") is False
    assert is_wake("Bé Xinh nghe đây") is False


def test_command_can_follow_wake_in_same_utterance() -> None:
    assert strip_wake_command("Bé Xinh ơi, mấy giờ rồi") == "may gio roi"


def test_identity_answer_uses_one_persona() -> None:
    answer = fast_answer("bạn là ai")
    assert answer is not None
    assert "Bé Xinh" in answer
    assert "Hà Linh" not in answer
