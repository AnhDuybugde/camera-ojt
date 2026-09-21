"""Test face-provisional merge: GID unknown gop vao GID da bind cung employee.

Khong ha nguong ReID; chi dung streak face-match lap lai (chiu flicker).
"""
from types import SimpleNamespace

from scripts.run_workstate import (
    _candidate_vote_reached,
    _merge_unknown_gid_by_face,
)


class _FakeManager:
    def __init__(self, bound: dict[int, str | None]) -> None:
        self.identities = {
            gid: SimpleNamespace(employee_id=emp) for gid, emp in bound.items()
        }
        self.merged: list[tuple[int, int]] = []

    def employee_id_of(self, gid: int) -> str | None:
        record = self.identities.get(gid)
        return record.employee_id if record is not None else None

    def merge_identity(self, duplicate_gid: int, canonical_gid: int) -> bool:
        dup = self.identities.get(duplicate_gid)
        canon = self.identities.get(canonical_gid)
        if dup is None or canon is None:
            return False
        if dup.employee_id and canon.employee_id and dup.employee_id != canon.employee_id:
            return False
        if canon.employee_id is None:
            canon.employee_id = dup.employee_id
        del self.identities[duplicate_gid]
        self.merged.append((duplicate_gid, canonical_gid))
        return True


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict]] = []

    def push(self, kind: str, payload: dict) -> None:
        self.items.append((kind, payload))


class _FakeDayCache:
    def status_of(self, day_str: str, gid: int):  # noqa: ANN001, ANN202
        return None


def test_streak_needs_required_hits() -> None:
    streaks: dict[int, list] = {}
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 0.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 1.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 2.0) == "1"
    assert 2 not in streaks  # reset sau khi fire


def test_streak_tolerates_unknown_flicker() -> None:
    streaks: dict[int, list] = {}
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 0.0) is None
    assert _candidate_vote_reached(streaks, 2, None, 0.0, 1.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.5, 2.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.9, 3.0) == "1"


def test_streak_resets_on_other_employee_and_low_score() -> None:
    streaks: dict[int, list] = {}
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 0.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 1.0) is None
    # Employee khac -> reset, ke ca da 2 hits.
    assert _candidate_vote_reached(streaks, 2, "2", 0.9, 2.0) is None
    assert _candidate_vote_reached(streaks, 2, "2", 0.9, 3.0) is None
    assert _candidate_vote_reached(streaks, 2, "2", 0.9, 4.0) == "2"
    # Score duoi min_score khong dem.
    streaks.clear()
    for _ in range(5):
        assert _candidate_vote_reached(streaks, 3, "1", 0.10, 0.0) is None


def test_streak_expires_after_window() -> None:
    streaks: dict[int, list] = {}
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 0.0, window_s=10.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 1.0, window_s=10.0) is None
    # Cach 20s -> het han, dem lai tu 1.
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 21.0, window_s=10.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 22.0, window_s=10.0) is None
    assert _candidate_vote_reached(streaks, 2, "1", 0.8, 23.0, window_s=10.0) == "1"


def _merge_setup():
    manager = _FakeManager({1: "1", 2: None})
    gid_alias: dict[int, int] = {}
    gid_to_person = {1: "1"}
    gid_to_display = {1: "Anh Duy"}
    candidates: dict[int, list] = {2: [("Anh Duy", 0.78)]}
    queue = _FakeQueue()
    return manager, gid_alias, gid_to_person, gid_to_display, candidates, queue


def test_merge_unknown_into_bound() -> None:
    manager, alias, persons, displays, candidates, queue = _merge_setup()
    ok = _merge_unknown_gid_by_face(
        manager, alias, persons, displays, candidates,
        _FakeDayCache(), queue, object(), (), "2026-09-16",
        2, "1", "Anh Duy",
    )
    assert ok is True
    assert alias == {2: 1}
    assert manager.merged == [(2, 1)]
    assert persons[2] == "1" and displays[2] == "Anh Duy"
    assert candidates[1] == [("Anh Duy", 0.78)]
    assert queue.items and queue.items[0][0] == "room_status"
    assert queue.items[0][1]["merged_into"] == 1


def test_merge_refuses_when_no_bound_conflict() -> None:
    manager = _FakeManager({2: None})
    ok = _merge_unknown_gid_by_face(
        manager, {}, {}, {}, {}, _FakeDayCache(), _FakeQueue(),
        object(), (), "2026-09-16", 2, "1", "Anh Duy",
    )
    assert ok is False


def test_merge_refuses_when_raw_already_bound_or_aliased() -> None:
    manager = _FakeManager({1: "1", 2: "2"})
    assert _merge_unknown_gid_by_face(
        manager, {}, {2: "2"}, {}, {}, _FakeDayCache(), _FakeQueue(),
        object(), (), "2026-09-16", 2, "1", "Anh Duy",
    ) is False
    manager2 = _FakeManager({1: "1", 2: None})
    assert _merge_unknown_gid_by_face(
        manager2, {2: 1}, {}, {}, {}, _FakeDayCache(), _FakeQueue(),
        object(), (), "2026-09-16", 2, "1", "Anh Duy",
    ) is False
    manager3 = _FakeManager({1: "1"})
    assert _merge_unknown_gid_by_face(
        manager3, {}, {}, {}, {}, _FakeDayCache(), _FakeQueue(),
        object(), (), "2026-09-16", 9, "1", "Anh Duy",
    ) is False
