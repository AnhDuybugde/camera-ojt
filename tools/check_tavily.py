"""One-request live Tavily probe; prints only provider, count and latency."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    from camera_tracking.voice.qa_tools import _search_tavily
    started = time.monotonic()
    results = _search_tavily("Python official documentation", 3)
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if results is None:
        print("Tavily is not configured.")
        return 2
    print({"provider": "tavily", "status": "ok" if results else "empty",
           "result_count": len(results), "elapsed_ms": elapsed_ms})
    return 0 if results else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Tavily probe failed:", type(error).__name__)
        raise SystemExit(1)
