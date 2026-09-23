"""Apply the additive Supabase migration without printing credentials."""
import argparse
from pathlib import Path
from urllib.parse import urlsplit, unquote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    from dotenv import dotenv_values
    import psycopg2
    root = Path(__file__).resolve().parents[1]
    parsed = urlsplit(dotenv_values(root / ".env")["CONNECT_STRING"])
    # Parse fields separately: raw reserved characters in existing passwords
    # must not be interpreted as libpq URI separators.
    connection = psycopg2.connect(host=parsed.hostname, port=parsed.port or 5432,
        user=unquote(parsed.username or ""), password=unquote(parsed.password or ""),
        dbname=parsed.path.lstrip("/") or "postgres", sslmode="require", connect_timeout=10)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if args.apply:
                cursor.execute((root / "migrations/supabase/001_unified_platform.sql").read_text())
            cursor.execute("SELECT count(*) FROM pg_tables WHERE schemaname='public' "
                           "AND tablename IN ('camera_events','camera_records','camera_memberships') AND rowsecurity")
            print("Platform tables with RLS:", cursor.fetchone()[0])
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("Cloud migration failed:", type(error).__name__)
        raise SystemExit(1)
