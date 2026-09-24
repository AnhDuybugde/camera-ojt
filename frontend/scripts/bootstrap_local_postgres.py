"""Initialize a fresh local PostgreSQL cluster for AI Mind without source secrets.

Requires PG_SUPERUSER_PASSWORD in the process environment. Creates one
application role/database and rotates the installer superuser password. The
generated credentials are stored only in the current user's LocalAppData.
Run --activate only after SQLite migration and PostgreSQL checks succeed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
from urllib.parse import quote

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
SECRET_FILE = Path(os.environ.get("LOCALAPPDATA", str(ROOT / ".runtime-temp"))) / "AI-Mind" / "postgres-credentials.json"
ROLE = "ai_mind_app"
DATABASE = "ai_mind_attendance"


def _url(password: str) -> str:
    return f"postgresql+psycopg://{ROLE}:{quote(password, safe='')}@127.0.0.1:5432/{DATABASE}"


def bootstrap() -> None:
    initial = os.environ.get("PG_SUPERUSER_PASSWORD", "")
    if not initial:
        raise ValueError("Set PG_SUPERUSER_PASSWORD for this process; do not pass it as an argument")
    if SECRET_FILE.exists():
        raise RuntimeError(f"Credentials file already exists: {SECRET_FILE}; refusing to rotate or overwrite")
    with psycopg.connect(host="127.0.0.1", port=5432, dbname="postgres", user="postgres", password=initial, autocommit=True) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
            names = {row[0] for row in cursor.fetchall()}
            if names != {"postgres"}:
                raise RuntimeError(f"Cluster is not fresh ({names}); refusing automatic bootstrap")
            cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (ROLE,))
            if cursor.fetchone():
                raise RuntimeError("Application role already exists; refusing automatic bootstrap")

            super_password = secrets.token_urlsafe(48)
            app_password = secrets.token_urlsafe(48)
            SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {"superuser": "postgres", "superuser_password": super_password,
                       "application_role": ROLE, "database": DATABASE, "database_url": _url(app_password)}
            temporary = SECRET_FILE.with_suffix(".tmp")
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream)
            os.replace(temporary, SECRET_FILE)

            cursor.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                sql.Identifier("postgres"), sql.Literal(super_password)))
            cursor.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(ROLE), sql.Literal(app_password)))
            cursor.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(DATABASE), sql.Identifier(ROLE)))
    print(f"PostgreSQL role/database created. Credentials saved at {SECRET_FILE} (values not displayed).")


def activate() -> None:
    credentials = json.loads(SECRET_FILE.read_text(encoding="utf-8"))
    url = credentials["database_url"]
    config = ROOT / ".env"
    if not config.is_file():
        raise FileNotFoundError(config)
    original = config.read_text(encoding="utf-8")
    backup_dir = ROOT / ".runtime-temp"
    backup_dir.mkdir(exist_ok=True)
    shutil.copy2(config, backup_dir / f"env_before_postgres_{datetime.now():%Y%m%d_%H%M%S}.txt")
    lines = [line for line in original.splitlines() if not line.startswith("DATABASE_URL=")]
    lines.append(f"DATABASE_URL={url}")
    temporary = config.with_name(".env.postgres.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, config)
    print("DATABASE_URL activated in .env. Restart the application to use PostgreSQL.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activate", action="store_true", help="write DATABASE_URL to .env after migration checks")
    args = parser.parse_args()
    if args.activate:
        activate()
    else:
        bootstrap()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
