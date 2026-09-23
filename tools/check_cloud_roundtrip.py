"""Exercise CAS sync with isolated synthetic data, then remove the exact row."""
from pathlib import Path
import json
import sys
import uuid
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dotenv import load_dotenv, dotenv_values
    load_dotenv(ROOT / ".env")
    import psycopg2
    from camera_tracking.store.cloud import CloudEventWriter

    key = "codex-roundtrip-" + str(uuid.uuid4())
    writer = CloudEventWriter.from_env()
    try:
        first = {"kind": "audit_logs", "record_key": key,
                 "payload": json.dumps({"probe": True, "version": 1}),
                 "deleted": 0, "base_revision": 0}
        inserted = writer.compare_batch([first])[0]
        assert not inserted["conflict"] and inserted["record"]["revision"] == 1
        duplicate = writer.compare_batch([first])[0]
        assert not duplicate["conflict"] and duplicate["record"]["revision"] == 1
        second = {**first, "payload": json.dumps({"probe": True, "version": 2}),
                  "base_revision": 1}
        updated = writer.compare_batch([second])[0]
        assert not updated["conflict"] and updated["record"]["revision"] == 2
        stale = {**second, "payload": json.dumps({"probe": True, "version": 3})}
        conflict = writer.compare_batch([stale])[0]
        assert conflict["conflict"] and conflict["record"]["revision"] == 2
        print({"provider": "supabase", "status": "ok", "checks": 4,
               "revision": updated["record"]["revision"], "cleanup": "complete"})
    finally:
        parsed = urlsplit(dotenv_values(ROOT / ".env")["CONNECT_STRING"])
        connection = psycopg2.connect(host=parsed.hostname, port=parsed.port or 5432,
            user=unquote(parsed.username or ""), password=unquote(parsed.password or ""),
            dbname=parsed.path.lstrip("/") or "postgres", sslmode="require", connect_timeout=10)
        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM public.camera_records WHERE kind=%s AND record_key=%s",
                                   ("audit_logs", key))
        finally:
            connection.close()


if __name__ == "__main__":
    main()
