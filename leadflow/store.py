"""SQLite store for triaged submissions and routing decisions.

submissions holds one row per submission_id, replaced on replay. The duplicate
check reads it. routing_decisions appends a row each time an existing-account
decision is saved, so replaying the same lead keeps each run.
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from leadflow.models import TIMESTAMP_FORMAT, Lead, OwnershipDecision, PriorSubmission, TriageResult

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "leadflow.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    submitted_at TEXT NOT NULL,
    work_email TEXT NOT NULL,
    domain TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    assigned_rep_id TEXT
);
CREATE INDEX IF NOT EXISTS submissions_email ON submissions (work_email);
CREATE INDEX IF NOT EXISTS submissions_domain ON submissions (domain);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    path TEXT NOT NULL,
    outcome TEXT NOT NULL,
    assigned_rep_id TEXT,
    candidate_rep_ids TEXT NOT NULL,
    account_ids TEXT NOT NULL,
    reason TEXT NOT NULL,
    brief TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS routing_decisions_submission ON routing_decisions (submission_id);
"""


class DecisionStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def record_submission(self, lead: Lead, result: TriageResult) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO submissions (submission_id, submitted_at, work_email, domain, outcome,"
                " reason, assigned_rep_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    lead.submission_id,
                    lead.submitted_at,
                    lead.work_email,
                    lead.email_domain,
                    result.outcome,
                    result.reason,
                    result.assigned_rep_id,
                ),
            )

    def prior_submissions(self, lead: Lead) -> List[PriorSubmission]:
        """Earlier submissions sharing the lead's email or email domain."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT submission_id, submitted_at, work_email, domain, outcome, assigned_rep_id FROM submissions"
                " WHERE (work_email = ? OR domain = ?) AND submission_id != ?",
                (lead.work_email, lead.email_domain, lead.submission_id),
            ).fetchall()
        return [PriorSubmission(**dict(row)) for row in rows]

    def submission(self, submission_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM submissions WHERE submission_id = ?", (submission_id,)).fetchone()
        return dict(row) if row else None

    def save(self, decision: OwnershipDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO routing_decisions (submission_id, decided_at, path, outcome, assigned_rep_id,"
                " candidate_rep_ids, account_ids, reason, brief) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision.submission_id,
                    datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT),
                    "existing_account",
                    decision.outcome,
                    decision.assigned_rep_id,
                    json.dumps(list(decision.candidate_rep_ids)),
                    json.dumps(list(decision.account_ids)),
                    decision.reason,
                    decision.brief,
                ),
            )

    def latest(self, submission_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM routing_decisions WHERE submission_id = ? ORDER BY id DESC LIMIT 1",
                (submission_id,),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["candidate_rep_ids"] = json.loads(record["candidate_rep_ids"])
        record["account_ids"] = json.loads(record["account_ids"])
        return record

    def count(self, submission_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM routing_decisions WHERE submission_id = ?", (submission_id,)
            ).fetchone()[0]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()
