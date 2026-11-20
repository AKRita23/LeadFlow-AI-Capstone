"""SQLite store for triaged submissions, routing decisions, and assignments.

submissions holds one row per submission_id, replaced on replay. The duplicate
check reads it. routing_decisions appends a row each time an existing-account
decision is saved, so replaying the same lead keeps each run. assignments
holds one row per committed new prospect; with the workbook's counts as the
baseline, it gives each rep's live load for the day and week. handoffs is the
mock Slack outbox: one row per message sent to a rep or to RevOps. status_events
is each assigned lead's status history (New, Working, Converted, Disqualified).
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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

CREATE TABLE IF NOT EXISTS assignments (
    submission_id TEXT PRIMARY KEY,
    committed_at TEXT NOT NULL,
    day TEXT NOT NULL,
    week TEXT NOT NULL,
    outcome TEXT NOT NULL,
    rep_id TEXT,
    backup_rep_id TEXT,
    skipped TEXT NOT NULL,
    intent_points INTEGER NOT NULL,
    urgency_tier TEXT NOT NULL,
    method TEXT NOT NULL,
    rationale TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assignments_rep_day ON assignments (rep_id, day);
CREATE INDEX IF NOT EXISTS assignments_rep_week ON assignments (rep_id, week);

CREATE TABLE IF NOT EXISTS handoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL,
    sent_at TEXT NOT NULL,
    recipient TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    UNIQUE (submission_id, kind, recipient)
);

CREATE TABLE IF NOT EXISTS status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    actor TEXT NOT NULL,
    rep_id TEXT,
    note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS status_events_submission ON status_events (submission_id);
"""

ASSIGNMENT_COLUMNS = (
    "submission_id", "committed_at", "day", "week", "outcome", "rep_id", "backup_rep_id", "skipped",
    "intent_points", "urgency_tier", "method", "rationale",
)


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

    def rep_load(
        self, day: str, week: str, exclude_submission_id: Optional[str] = None, conn: Optional[sqlite3.Connection] = None
    ) -> Dict[str, Tuple[int, int]]:
        """Leads committed to each rep on day, and intent points committed in week."""
        query = (
            "SELECT rep_id, SUM(day = ?) AS today, SUM(CASE WHEN week = ? THEN intent_points ELSE 0 END) AS points"
            " FROM assignments WHERE rep_id IS NOT NULL AND submission_id != ? GROUP BY rep_id"
        )
        params = (day, week, exclude_submission_id or "")
        if conn is not None:
            rows = conn.execute(query, params).fetchall()
        else:
            with self._connect() as own:
                rows = own.execute(query, params).fetchall()
        return {row["rep_id"]: (row["today"] or 0, row["points"] or 0) for row in rows}

    def assignment(self, submission_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
        query, params = "SELECT * FROM assignments WHERE submission_id = ?", (submission_id,)
        if conn is not None:
            row = conn.execute(query, params).fetchone()
        else:
            with self._connect() as own:
                row = own.execute(query, params).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["skipped"] = json.loads(record["skipped"])
        return record

    def insert_assignment(self, conn: sqlite3.Connection, record: dict) -> None:
        """Save an assignment and set the submission's assigned rep. Call inside locked()."""
        values = {**record, "skipped": json.dumps(record["skipped"])}
        conn.execute(
            f"INSERT INTO assignments ({', '.join(ASSIGNMENT_COLUMNS)})"
            f" VALUES ({', '.join('?' for _ in ASSIGNMENT_COLUMNS)})",
            tuple(values[column] for column in ASSIGNMENT_COLUMNS),
        )
        self.set_assigned_rep(conn, record["submission_id"], record["rep_id"])

    def set_assigned_rep(self, conn: sqlite3.Connection, submission_id: str, rep_id: Optional[str]) -> None:
        conn.execute("UPDATE submissions SET assigned_rep_id = ? WHERE submission_id = ?", (rep_id, submission_id))

    def add_handoff(
        self, conn: sqlite3.Connection, submission_id: str, sent_at: str, recipient: str, kind: str,
        subject: str, body: str,
    ) -> bool:
        """Record a message in the outbox. Returns False if this message was already sent."""
        cursor = conn.execute(
            "INSERT OR IGNORE INTO handoffs (submission_id, sent_at, recipient, kind, subject, body)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (submission_id, sent_at, recipient, kind, subject, body),
        )
        return cursor.rowcount == 1

    def handoffs(
        self, submission_id: Optional[str] = None, limit: int = 50, conn: Optional[sqlite3.Connection] = None
    ) -> List[dict]:
        """Outbox messages, newest first, optionally for one submission."""
        where, params = ("WHERE submission_id = ?", (submission_id,)) if submission_id else ("", ())
        query = f"SELECT * FROM handoffs {where} ORDER BY id DESC LIMIT ?"
        if conn is not None:
            rows = conn.execute(query, (*params, limit)).fetchall()
        else:
            with self._connect() as own:
                rows = own.execute(query, (*params, limit)).fetchall()
        return [dict(row) for row in rows]

    def add_status(
        self, conn: sqlite3.Connection, submission_id: str, changed_at: str, status: str, actor: str,
        rep_id: Optional[str], note: str,
    ) -> None:
        conn.execute(
            "INSERT INTO status_events (submission_id, changed_at, status, actor, rep_id, note) VALUES (?, ?, ?, ?, ?, ?)",
            (submission_id, changed_at, status, actor, rep_id, note),
        )

    def status_history(self, submission_id: str, conn: Optional[sqlite3.Connection] = None) -> List[dict]:
        """Status events for a submission, oldest first."""
        query, params = "SELECT * FROM status_events WHERE submission_id = ? ORDER BY id", (submission_id,)
        if conn is not None:
            rows = conn.execute(query, params).fetchall()
        else:
            with self._connect() as own:
                rows = own.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def assigned_leads(self, conn: sqlite3.Connection) -> List[dict]:
        """Assignments that currently have a rep, for the watchdog."""
        return [dict(row) for row in conn.execute("SELECT * FROM assignments WHERE outcome = 'assigned'").fetchall()]

    def reassign(self, conn: sqlite3.Connection, submission_id: str, rep_id: Optional[str]) -> None:
        """Move an assignment to rep_id, or to the RevOps queue when rep_id is None. Call inside locked()."""
        outcome = "assigned" if rep_id else "revops_queue"
        conn.execute(
            "UPDATE assignments SET rep_id = ?, backup_rep_id = NULL, outcome = ? WHERE submission_id = ?",
            (rep_id, outcome, submission_id),
        )
        self.set_assigned_rep(conn, submission_id, rep_id)

    def clear_assignments(self) -> int:
        """Delete every assignment, handoff, and status event, and unset assigned reps. For resetting the demo."""
        with self._connect() as conn:
            deleted = conn.execute("DELETE FROM assignments").rowcount
            conn.execute("DELETE FROM handoffs")
            conn.execute("DELETE FROM status_events")
            conn.execute("UPDATE submissions SET assigned_rep_id = NULL WHERE outcome = 'new_prospect'")
        return deleted

    @contextmanager
    def locked(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that holds the database lock from the first read.

        Two commits for the same rep cannot both read a free slot: the second
        waits until the first has written its assignment.
        """
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()
