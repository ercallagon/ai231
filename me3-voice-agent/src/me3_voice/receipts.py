"""Private opt-in durable dispatch/receipt ledger; pending effects are never retried."""

import hashlib
import json
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .async_utils import await_thread_future


class DispatchUncertain(RuntimeError):
    task_status = "unknown"
    user_summary = "Dispatch ended without a confirmed outcome. It was not retried; inspect the target before trying again."


class ReceiptStore:
    def __init__(self, path):
        self.path = Path(path).absolute()
        allowed = Path.cwd().resolve() / "var/cache"
        if self.path.is_symlink() or not self.path.resolve().is_relative_to(allowed):
            raise ValueError("receipt store must be under project var/cache")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("receipt store must be a regular file")
        finally:
            os.close(fd)
        if self.path.stat().st_mode & 0o077:
            raise ValueError("receipt store must be private (0600)")
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swift-receipts")

    def _transaction(self, key, fingerprint, result=None):
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT)"
            )
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT fingerprint,result FROM receipts WHERE id=?", (key,)
            ).fetchone()
            if row and row[0] != fingerprint:
                raise ValueError("action identity reused with different arguments")
            if result is not None:
                db.execute(
                    "UPDATE receipts SET result=? WHERE id=? AND fingerprint=?",
                    (json.dumps(result, allow_nan=False), key, fingerprint),
                )
                return None
            if row:
                return ("pending", None) if row[1] is None else ("complete", json.loads(row[1]))
            if db.execute("SELECT COUNT(*) FROM receipts").fetchone()[0] >= 10000:
                raise ValueError("durable receipt capacity reached; reconcile/archive explicitly")
            db.execute("INSERT INTO receipts VALUES (?,?,NULL)", (key, fingerprint))
            return "new", None

    async def claim(self, key, call, context=None):
        fingerprint = hashlib.sha256(
            json.dumps(
                {"tool": call.name, "arguments": call.arguments, "target": context},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        state, result = await await_thread_future(
            self.worker.submit(self._transaction, key, fingerprint)
        )
        return state, result, fingerprint

    async def finish(self, key, fingerprint, result):
        await await_thread_future(self.worker.submit(self._transaction, key, fingerprint, result))

    async def close(self):
        # All executor jobs must be settled before closing this owned worker.
        self.worker.shutdown(wait=True)


def main():
    """Read-only local recovery inspection; never opens an adapter or submits work."""
    import argparse
    from urllib.parse import quote

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("command", choices=["inspect"])
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    path = args.store.absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(Path.cwd().resolve() / "var/cache")
        or path.stat().st_mode & 0o077
    ):
        parser.error("select an existing private store under project var/cache")
    with sqlite3.connect("file:" + quote(str(path)) + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id,result FROM receipts ORDER BY rowid DESC LIMIT 20").fetchall()
    print(
        json.dumps(
            [
                {
                    "operation_id": key,
                    "state": "recorded" if value else "unknown",
                    "summary": json.loads(value).get("summary", "Recorded result")
                    if value
                    else "Dispatch was recorded without a final receipt; do not replay automatically.",
                }
                for key, value in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
