"""Cooperative cancel flags for in-process fine-tune jobs (gateway thread)."""

from __future__ import annotations

import threading
from typing import Optional

_CANCEL_EVENTS: dict[str, threading.Event] = {}
_LOCK = threading.Lock()


def clear_cancel(job_id: str) -> None:
    with _LOCK:
        _CANCEL_EVENTS.pop(job_id, None)


def request_cancel(job_id: str) -> None:
    with _LOCK:
        ev = _CANCEL_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _CANCEL_EVENTS[job_id] = ev
        ev.set()


def ensure_cancel_event(job_id: str) -> threading.Event:
    with _LOCK:
        ev = _CANCEL_EVENTS.get(job_id)
        if ev is None:
            ev = threading.Event()
            _CANCEL_EVENTS[job_id] = ev
        return ev


def is_cancel_requested(job_id: Optional[str] = None) -> bool:
    import os

    jid = job_id or os.environ.get("FTAAS_JOB_ID", "")
    if not jid:
        return False
    with _LOCK:
        ev = _CANCEL_EVENTS.get(jid)
    return bool(ev and ev.is_set())


class JobCancelled(Exception):
    """Raised by the local runner when a user stops training."""
