"""Kiem tra tool web_search (search ngoai cho Gemini Free Tier)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from camera_tracking.voice import qa_tools as qa


def test_web_search_registered():
    names = [d["name"] for d in qa.TOOL_DECLARATIONS]
    assert "web_search" in names
    assert callable(qa.TOOL_FUNCS.get("web_search"))
    catalog = qa.build_tool_catalog()
    assert "web_search(query" in catalog


def test_search_web_raw_validates_query():
    import pytest

    with pytest.raises(RuntimeError):
        qa.search_web_raw("")
    with pytest.raises(RuntimeError):
        qa.search_web_raw("a")


def test_search_web_raw_fallback_no_keys(monkeypatch):
    for key in ("TAVILY_API_KEY", "SERPER_API_KEY", "BRAVE_API_KEY",
                "EXA_API_KEY", "SEARXNG_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        qa, "_search_free_fallback",
        lambda q, n: [{"title": "T", "url": "http://x",
                       "content": "C" * 10}])
    hits = qa.search_web_raw("Bitcoin price today", 3)
    assert isinstance(hits, list) and hits
    assert set(hits[0]) >= {"title", "url", "content"}


def test_provider_chain_prefers_tavily(monkeypatch):
    for key in ("SERPER_API_KEY", "BRAVE_API_KEY", "EXA_API_KEY",
                "SEARXNG_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(
        qa, "_http_post_json",
        lambda url, payload, headers=None, timeout=12.0: {
            "results": [{"title": "BTC", "url": "http://x",
                         "content": "giá bitcoin"}]})
    hits = qa.search_web_raw("Bitcoin price", 2)
    assert hits[0]["title"] == "BTC"


def test_build_search_context_truncates():
    results = [{"title": f"T{i}", "url": f"http://x/{i}",
                "content": "nội dung " * 300} for i in range(5)]
    ctx = qa.build_search_context(results, 1000)
    assert len(ctx) <= 1005
    assert "[1]" in ctx


def test_web_search_spoken_fallback(monkeypatch):
    monkeypatch.setattr(
        qa, "search_web_raw",
        lambda q, n=5: [{"title": "BTC", "url": "http://x",
                         "content": "giá 100k"}])
    text = qa.web_search("Bitcoin price today USD", 3)
    assert "BTC" in text
    assert len(text) <= 1500
