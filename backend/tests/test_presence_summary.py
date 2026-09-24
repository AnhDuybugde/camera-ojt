from camera_tracking.runtime.presence import summarize_presence


def test_presence_summary_separates_visible_fresh_and_stale() -> None:
    summary = summarize_presence(
        [
            {"gid": 1, "in_room": True, "visible": True, "last_seen_ago_s": 0},
            {"gid": 2, "in_room": True, "visible": False, "last_seen_ago_s": 40},
            {"gid": 3, "in_room": True, "visible": False, "last_seen_ago_s": 500},
            {"gid": 4, "in_room": False, "visible": False, "last_seen_ago_s": 1},
        ],
        stale_after_s=120,
    )
    assert summary.visible_count == 1
    assert summary.active_count == 2
    assert summary.logical_count == 3
    assert summary.stale_count == 1


def test_presence_summary_deduplicates_aliases() -> None:
    summary = summarize_presence(
        [
            {"gid": 7, "in_room": True, "visible": True, "last_seen_ago_s": 0},
            {"gid": 9, "in_room": True, "visible": False, "last_seen_ago_s": 5},
        ],
        aliases={9: 7},
        stale_after_s=30,
    )
    assert summary.visible_count == 1
    assert summary.active_count == 1
    assert summary.logical_count == 1
    assert summary.stale_count == 0
