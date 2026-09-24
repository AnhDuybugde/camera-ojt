"""Deterministic, cache-friendly phrase library for semantic office events."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from camera_tracking.audio.events import AudioEventKind


PHRASE_POOLS: dict[str, tuple[str, ...]] = {
    "door_enter_known": (
        "{name} tới rồi nè! Bé Xinh chào nha.",
        "Chào {name} nha, Bé Xinh thấy rồi.",
        "Hello {name}, chúc một ngày làm việc vui vẻ nha.",
    ),
    "door_enter_generic": (
        "Chào nha! Bé Xinh thấy bạn rồi.",
        "Hello, chào bạn nha!",
        "Chào bạn, chúc một ngày thật vui nha.",
    ),
    "door_enter_group": (
        "Chào mọi người nha! Bé Xinh thấy hết rồi.",
        "Hello mọi người, chúc cả nhà làm việc vui vẻ nha.",
    ),
    "door_exit_known": (
        "{name} đi đâu đó? Nhớ quay lại nha.",
        "Đi đâu vậy {name}? Bé Xinh ở đây chờ nha.",
        "{name} ra ngoài hả? Đi cẩn thận nha.",
    ),
    "door_exit_generic": (
        "Đi đâu đó? Nhớ quay lại nha.",
        "Ra ngoài hả? Đi cẩn thận nha.",
    ),
    "be_xinh_near_known": (
        "Ủa, {name} qua thăm Bé Xinh hả?",
        "Hello {name}, lại gần Bé Xinh rồi nha.",
        "{name} tới gần rồi, Bé Xinh xin chào nha.",
    ),
    "be_xinh_near_generic": (
        "Ủa, lại gần Bé Xinh hả?",
        "Hello, qua thăm Bé Xinh hả?",
    ),
    "water_known": (
        "{name} ơi, nhớ uống nước nha.",
        "Làm việc chăm quá rồi đó {name}, uống chút nước đi nha.",
        "Nạp nước cho CPU con người một chút nào {name}.",
    ),
    "water_generic": (
        "Nè, nhớ uống nước nha!",
        "Làm việc chăm quá rồi, uống chút nước nha.",
        "Nạp nước cho CPU con người một chút nào.",
    ),
    # Privacy boundary: this pool must never contain a name placeholder.
    "restroom_generic": (
        "Khu vực riêng tư, Bé Xinh xin phép giữ im lặng nha.",
        "Bé Xinh quay mặt chỗ khác nha.",
        "Yên tâm, Bé Xinh không nhắc tên ở khu vực này đâu nha.",
    ),
    "wave_known": (
        "Hihi, {name} chào Bé Xinh hả? Chào nha.",
        "Bé Xinh thấy {name} vẫy tay rồi nha!",
    ),
    "wave_generic": (
        "Hihi, Bé Xinh thấy rồi nha! Chào bạn.",
        "Bé Xinh thấy bạn vẫy tay rồi nha!",
    ),
}


@dataclass(frozen=True, slots=True)
class PhraseSelection:
    text: str
    fallback_text: str
    pool_key: str


def _pick(pool: tuple[str, ...], seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).digest()
    return pool[int.from_bytes(digest[:4], "big") % len(pool)] if pool else ""


def _keys(kind: AudioEventKind, known: bool) -> tuple[str, str]:
    if kind is AudioEventKind.DOOR_ENTER:
        return ("door_enter_known" if known else "door_enter_generic", "door_enter_generic")
    if kind is AudioEventKind.DOOR_EXIT:
        return ("door_exit_known" if known else "door_exit_generic", "door_exit_generic")
    if kind is AudioEventKind.BE_XINH_NEAR:
        return ("be_xinh_near_known" if known else "be_xinh_near_generic", "be_xinh_near_generic")
    if kind is AudioEventKind.WATER:
        return ("water_known" if known else "water_generic", "water_generic")
    if kind is AudioEventKind.RESTROOM:
        return ("restroom_generic", "restroom_generic")
    if kind is AudioEventKind.WAVE:
        return ("wave_known" if known else "wave_generic", "wave_generic")
    raise KeyError(kind)


def select_phrase(
    kind: AudioEventKind,
    *,
    event_id: str,
    display_name: str = "",
    allow_name: bool = True,
) -> PhraseSelection:
    name = " ".join(str(display_name or "").split())[:80]
    known = bool(name and allow_name)
    pool_key, fallback_key = _keys(kind, known)
    text = _pick(PHRASE_POOLS[pool_key], f"{event_id}:{pool_key}")
    fallback = _pick(PHRASE_POOLS[fallback_key], f"{kind.value}:fallback")
    if known:
        text = text.format(name=name)
    return PhraseSelection(text=text, fallback_text=fallback, pool_key=pool_key)


def select_group_arrival_phrase(event_ids: Iterable[str]) -> PhraseSelection:
    seed = "|".join(sorted(str(value) for value in event_ids))
    text = _pick(PHRASE_POOLS["door_enter_group"], f"group:{seed}")
    return PhraseSelection(
        text=text,
        fallback_text=PHRASE_POOLS["door_enter_generic"][0],
        pool_key="door_enter_group",
    )


def iter_prewarm_texts(
    people: Iterable[tuple[str, str]],
    *,
    critical_only: bool = False,
) -> list[str]:
    texts: list[str] = []
    selected = {
        key: pool
        for key, pool in PHRASE_POOLS.items()
        if not critical_only
        or key.startswith(("wave_", "door_", "be_xinh_"))
    }
    for key, pool in selected.items():
        if not key.endswith("_known"):
            texts.extend(pool)
    known_pools = [pool for key, pool in selected.items() if key.endswith("_known")]
    for _person_id, raw_name in people:
        name = " ".join(str(raw_name or "").split())[:80]
        if name:
            for pool in known_pools:
                texts.extend(template.format(name=name) for template in pool)
    return list(dict.fromkeys(texts))


__all__ = [
    "PHRASE_POOLS",
    "PhraseSelection",
    "iter_prewarm_texts",
    "select_group_arrival_phrase",
    "select_phrase",
]
