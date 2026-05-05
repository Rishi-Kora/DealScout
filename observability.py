"""Observability layer for DealScout (M5+).

Two distinct concerns live here:

  1. **Scratchpad** — mutable shared state. The team's findings-so-far. Each
     specialist reads it; some write to it. *What the team knows.*

  2. **TraceLogger** — append-only event log. Every agent start/finish, every
     tool call, every decision. JSONL on disk + generator interface for the
     post-M7 Streamlit UI to stream live. *What happened in what order.*

Both are designed to be thread-safe (specialists run in parallel from M3 on)
and to flush to disk immediately (so a partial run is still observable).

The `TraceEvent` Pydantic model below is the on-the-wire shape. Every event
written to trace.jsonl is one of these.
"""

import json
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """All trace event types in DealScout. Add sparingly."""

    AGENT_STARTED = "agent_started"
    AGENT_FINISHED = "agent_finished"
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    LLM_CALL = "llm_call"
    ERROR = "error"


class TraceEvent(BaseModel):
    """A single structured event in a run's trace.

    Written one-per-line to trace.jsonl. Contains everything needed to replay
    or render a run after the fact.
    """

    ts: str = Field(description="ISO 8601 timestamp of when the event happened.")
    source: str = Field(
        description="Which agent or component emitted the event "
                    "(e.g. 'coordinator', 'history_checker', 'tool:rc_lookup')."
    )
    event_type: EventType = Field(description="What kind of event this is.")
    payload: dict = Field(
        description="Event-specific data. Shape depends on event_type. Free-form by design."
    )


class Scratchpad:
    """Thread-safe append-only shared state for the team.

    Each entry records WHO posted WHAT KIND of finding, WHEN, and the VALUE.
    Specialists read the snapshot at the start of their work and (some) write
    findings at the end. Reads always see the current state; writes never
    overwrite earlier entries.

    Design note: this is intentionally a plain list-of-dicts internally, not
    a dict keyed by `key`. Two specialists can post the same `key` with
    different findings, and we want both visible — not one overwriting the
    other. The caller decides how to interpret duplicates.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._lock = threading.Lock()

    def add(self, source: str, key: str, value: Any) -> None:
        """Append one finding from `source` under `key` with `value`.

        Thread-safe. Returns nothing — entries are immutable once added.
        """
        with self._lock:
            self._entries.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                    "key": key,
                    "value": value,
                }
            )

    def snapshot(self) -> list[dict]:
        """Return a shallow copy of all entries in insertion order."""
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __bool__(self) -> bool:
        return len(self) > 0

    def to_user_message_section(self) -> str:
        """Render the scratchpad as text for inclusion in a specialist's prompt.

        Empty-state returns a clear "(scratchpad empty)" line so specialists
        know there's nothing yet, rather than seeing a blank section that
        could be misread as missing data.
        """
        snap = self.snapshot()
        if not snap:
            return "(scratchpad empty)"
        lines = []
        for entry in snap:
            lines.append(f"- [{entry['source']}] {entry['key']}: {entry['value']}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """For saving to output/<slug>/scratchpad.json."""
        return {"entries": self.snapshot()}


class TraceLogger:
    """Append-only event log for a DealScout run.

    Writes each event as a JSON line to trace.jsonl (immediate flush so
    a partial run is still observable on disk) AND keeps an in-memory
    list for end-of-run access.

    Thread-safe: parallel specialists may call `log()` concurrently.

    Streaming pattern (post-M7 Streamlit UI):
      Because each event is flushed to disk synchronously, a separate
      reader can tail trace.jsonl line by line and render events as
      they happen. No special API required from this class — file
      tailing is the live-stream contract.
    """

    def __init__(self, trace_path: Path | None = None) -> None:
        self._events: list[TraceEvent] = []
        self._lock = threading.Lock()
        self._trace_path = trace_path
        self._fh = None
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = trace_path.open("a", encoding="utf-8")

    def log(
        self,
        source: str,
        event_type: EventType,
        payload: dict | None = None,
    ) -> TraceEvent:
        """Build a TraceEvent and append it to memory and disk.

        Convenience wrapper — the common case. Returns the event for
        callers who want to inspect it (e.g. for in-line printing).
        """
        event = TraceEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            source=source,
            event_type=event_type,
            payload=payload or {},
        )
        self.log_event(event)
        return event

    def log_event(self, event: TraceEvent) -> None:
        """Append a pre-built event to memory and (if configured) to disk."""
        with self._lock:
            self._events.append(event)
            if self._fh is not None:
                self._fh.write(json.dumps(event.model_dump(mode="json")) + "\n")
                self._fh.flush()

    def events(self) -> list[TraceEvent]:
        """Return a snapshot of all events logged so far."""
        with self._lock:
            return list(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def close(self) -> None:
        """Close the underlying file handle (idempotent)."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "TraceLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
