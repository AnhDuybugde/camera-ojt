"""Safely copy both SQLite stores into an empty PostgreSQL database.

Stop the application/camera writers first. This script creates fresh SQLite
backups, validates their schema, copies all rows in one PostgreSQL transaction,
checks every value and row count, and never deletes or switches SQLite.
DATABASE_URL must be set in .env; do not pass passwords on the command line.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import os
from pathlib import Path
import sqlite3
import sys

from sqlalchemy import Date, DateTime, create_engine, func, select, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from database.postgres import MAIN_TABLES, SPATIAL_TABLES, metadata  # noqa: E402


def backup_sqlite(source: Path, folder: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.stem}_before_postgres_{datetime.now():%Y%m%d_%H%M%S_%f}.db"
    with sqlite3.connect(source) as original, sqlite3.connect(target) as copy:
        original.backup(copy)
        if copy.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"Backup integrity check failed: {source.name}")
    return target


def source_rows(path: Path, tables) -> dict[str, list[dict]]:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        found = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        expected = {table.name for table in tables}
        if found != expected:
            raise RuntimeError(f"Unexpected tables in {path.name}: missing={expected-found}, extra={found-expected}")
        result = {}
        for table in tables:
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table.name}")')}
            expected_columns = {column.name for column in table.columns}
            optional = {"presence_status", "temporary_checkout_at"} if table.name == "attendance" else set()
            if not expected_columns - optional <= columns or columns - expected_columns:
                raise RuntimeError(f"Column mismatch in {table.name}: missing={expected_columns-columns}, extra={columns-expected_columns}")
            result[table.name] = [dict(row) for row in connection.execute(f'SELECT * FROM "{table.name}"')]
        return result


def typed_rows(rows: list[dict], table) -> list[dict]:
    converted = []
    for row in rows:
        item = dict(row)
        if table.name == "attendance":
            item.setdefault("presence_status", "PRESENT" if item.get("check_in") else "ABSENT")
            item.setdefault("temporary_checkout_at", None)
        for column in table.columns:
            value = item[column.name]
            if isinstance(value, str) and isinstance(column.type, DateTime):
                item[column.name] = datetime.fromisoformat(value)
            elif isinstance(value, str) and isinstance(column.type, Date):
                item[column.name] = date.fromisoformat(value)
        converted.append(item)
    return converted


def migrate(main_path: Path, spatial_path: Path, database_url: str, backup_dir: Path):
    if not database_url.startswith("postgresql+psycopg://"):
        raise ValueError("DATABASE_URL must use postgresql+psycopg://")
    main_backup = backup_sqlite(main_path, backup_dir)
    spatial_backup = backup_sqlite(spatial_path, backup_dir)
    records = {
        **source_rows(main_backup, MAIN_TABLES),
        **source_rows(spatial_backup, SPATIAL_TABLES),
    }
    counts = {name: len(rows) for name, rows in records.items()}
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        metadata.create_all(engine)
        with engine.begin() as postgres:
            for table in (*MAIN_TABLES, *SPATIAL_TABLES):
                actual = postgres.scalar(select(func.count()).select_from(table))
                if actual:
                    raise RuntimeError(f"Destination {table.name} is not empty ({actual} rows)")

            for table in (*MAIN_TABLES, *SPATIAL_TABLES):
                rows = typed_rows(records[table.name], table)
                if rows:
                    postgres.execute(table.insert(), rows)
                actual = postgres.scalar(select(func.count()).select_from(table))
                if actual != counts[table.name]:
                    raise RuntimeError(f"Row count mismatch: {table.name} {actual} != {counts[table.name]}")

                # Compare all migrated values, including bcrypt hashes, blobs,
                # audit JSON, sync flags, spatial timestamps and original IDs.
                key = next(column.name for column in table.primary_key.columns)
                expected = {row[key]: row for row in rows}
                for row in postgres.execute(select(table)).mappings():
                    original = expected.pop(row[key])
                    for column in table.columns:
                        if row[column.name] != original[column.name]:
                            raise RuntimeError(f"Data mismatch in {table.name}.{column.name} at {row[key]}")
                if expected:
                    raise RuntimeError(f"Missing PostgreSQL rows in {table.name}: {len(expected)}")

                if "id" in table.c:
                    maximum = postgres.scalar(select(func.max(table.c.id)))
                    sequence = postgres.scalar(
                        text("SELECT pg_get_serial_sequence(:table_name, 'id')"),
                        {"table_name": table.name},
                    )
                    if sequence:
                        postgres.execute(text("SELECT setval(CAST(:sequence AS regclass), :value, :called)"), {
                            "sequence": sequence, "value": maximum or 1, "called": bool(maximum),
                        })
        return (main_backup, spatial_backup), counts
    finally:
        engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, default=settings.database_path)
    parser.add_argument("--spatial", type=Path, default=ROOT / "data" / "spatial_analytics.db")
    parser.add_argument("--backup-dir", type=Path, default=ROOT / "backup")
    parser.add_argument("--check-sqlite", action="store_true", help="validate source schemas without changing a database")
    args = parser.parse_args()
    if args.check_sqlite:
        for path, tables in ((args.main, MAIN_TABLES), (args.spatial, SPATIAL_TABLES)):
            for name, rows in source_rows(path, tables).items():
                print(f"{path.name}/{name}: {len(rows)}")
        return 0
    backups, counts = migrate(args.main, args.spatial, os.getenv("DATABASE_URL", ""), args.backup_dir)
    for path in backups:
        print(f"SQLite backup: {path}")
    for name, count in counts.items():
        print(f"{name}: {count} verified")
    print("Migration committed. Switch DATABASE_URL only after stopping SQLite writers and checking the application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
