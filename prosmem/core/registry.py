"""IntentionRegistry: store for active intentions (Interrupt Vector Table).

In-memory by default. Passing `db_path` turns on a thin SQLite persistence layer
so intentions survive across process restarts — the substrate for cross-session
prospective memory ("remind me next week"). The default (db_path=None) is pure
in-memory and byte-identical to the original behaviour, so all existing bench
runs are unaffected.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np

from prosmem.core.intention import (
    Intention,
    IntentionStatus,
    TriggerCondition,
    TriggerType,
)

# ---------------------------------------------------------------------------
# Full (loss-less) (de)serialization — unlike Intention.to_dict(), this round-
# trips embeddings + runtime state so a reloaded intention is immediately live.
# ---------------------------------------------------------------------------

def _emb(a: np.ndarray | None) -> list | None:
    return None if a is None else np.asarray(a, dtype=np.float32).tolist()


def _unemb(v) -> np.ndarray | None:
    return None if v is None else np.asarray(v, dtype=np.float32)


def intention_to_dict(it: Intention) -> dict:
    """Loss-less dict (embeddings + runtime fields included)."""
    tc = it.trigger_condition
    return {
        "intention_id": it.intention_id,
        "created_at": it.created_at,
        "source_context": it.source_context,
        "trigger_type": it.trigger_type.value,
        "action_description": it.action_description,
        "action_params": it.action_params,
        "priority": it.priority,
        "importance": it.importance,
        "expiry_step": it.expiry_step,
        "status": it.status.value,
        "implementation_intention": it.implementation_intention,
        "triggered_at_step": it.triggered_at_step,
        "completed_at_step": it.completed_at_step,
        "trigger_embedding": _emb(it.trigger_embedding),
        "trigger_condition": {
            "event_cues": tc.event_cues,
            "event_cue_embeddings": [_emb(e) for e in tc.event_cue_embeddings],
            "focality": tc.focality,
            "semantic_threshold": tc.semantic_threshold,
            "target_step": tc.target_step,
            "step_interval": tc.step_interval,
            "anchor_event": tc.anchor_event,
            "anchor_event_embedding": _emb(tc.anchor_event_embedding),
            "delay_steps": tc.delay_steps,
            "anchor_detected_step": tc.anchor_detected_step,
            "activity_completion": tc.activity_completion,
            "activity_embedding": _emb(tc.activity_embedding),
            "activity_completions": tc.activity_completions,
            "activity_completion_embeddings": [_emb(e) for e in tc.activity_completion_embeddings],
            "completed_activities": sorted(tc.completed_activities),
            "outcome": tc.outcome,
        },
    }


def intention_from_dict(d: dict) -> Intention:
    tcd = d["trigger_condition"]
    tc = TriggerCondition(
        event_cues=tcd["event_cues"],
        event_cue_embeddings=[_unemb(e) for e in tcd["event_cue_embeddings"]],
        focality=tcd["focality"],
        semantic_threshold=tcd["semantic_threshold"],
        target_step=tcd["target_step"],
        step_interval=tcd["step_interval"],
        anchor_event=tcd["anchor_event"],
        anchor_event_embedding=_unemb(tcd["anchor_event_embedding"]),
        delay_steps=tcd["delay_steps"],
        anchor_detected_step=tcd["anchor_detected_step"],
        activity_completion=tcd["activity_completion"],
        activity_embedding=_unemb(tcd["activity_embedding"]),
        activity_completions=tcd["activity_completions"],
        activity_completion_embeddings=[_unemb(e) for e in tcd["activity_completion_embeddings"]],
        completed_activities=set(tcd["completed_activities"]),
        outcome=tcd["outcome"],
    )
    return Intention(
        intention_id=d["intention_id"],
        created_at=d["created_at"],
        source_context=d["source_context"],
        trigger_type=TriggerType(d["trigger_type"]),
        trigger_condition=tc,
        action_description=d["action_description"],
        action_params=d["action_params"],
        priority=d["priority"],
        importance=d["importance"],
        expiry_step=d["expiry_step"],
        status=IntentionStatus(d["status"]),
        trigger_embedding=_unemb(d["trigger_embedding"]),
        implementation_intention=d["implementation_intention"],
        triggered_at_step=d["triggered_at_step"],
        completed_at_step=d["completed_at_step"],
    )


class IntentionRegistry:
    """Manages the lifecycle of all intentions. Analogous to an Interrupt Vector Table.

    Pass `db_path` to persist intentions to SQLite (cross-session PM). Default
    None → pure in-memory (original behaviour, used by all bench runs).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._intentions: dict[str, Intention] = {}
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        if db_path is not None:
            self._conn = sqlite3.connect(db_path)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS intentions ("
                "intention_id TEXT PRIMARY KEY, status TEXT, data TEXT)"
            )
            self._conn.commit()
            self._load()

    # ---- persistence helpers (no-ops when db disabled) ------------------
    def _load(self) -> None:
        assert self._conn is not None
        for (data,) in self._conn.execute("SELECT data FROM intentions"):
            it = intention_from_dict(json.loads(data))
            self._intentions[it.intention_id] = it

    def _persist_one(self, it: Intention) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO intentions (intention_id, status, data) VALUES (?,?,?)",
            (it.intention_id, it.status.value, json.dumps(intention_to_dict(it))),
        )
        self._conn.commit()

    def flush(self) -> None:
        """Persist the CURRENT state of every intention (call after status
        changes if you want them durable). No-op when db disabled."""
        if self._conn is None:
            return
        for it in self._intentions.values():
            self._persist_one(it)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- registry API ---------------------------------------------------
    def register(self, intention: Intention) -> str:
        """Add an intention and arm it. Returns intention_id."""
        self._intentions[intention.intention_id] = intention
        intention.arm()
        self._persist_one(intention)
        return intention.intention_id

    def get(self, intention_id: str) -> Intention | None:
        return self._intentions.get(intention_id)

    def remove(self, intention_id: str) -> None:
        self._intentions.pop(intention_id, None)
        if self._conn is not None:
            self._conn.execute("DELETE FROM intentions WHERE intention_id=?", (intention_id,))
            self._conn.commit()

    def get_armed(self) -> list[Intention]:
        """Return all intentions eligible for trigger checking."""
        return [i for i in self._intentions.values() if i.is_active()]

    def get_by_status(self, status: IntentionStatus) -> list[Intention]:
        return [i for i in self._intentions.values() if i.status == status]

    def expire_overdue(self, current_step: int) -> list[Intention]:
        """Mark expired intentions. Returns list of newly expired."""
        expired = []
        for i in self.get_armed():
            if i.is_expired(current_step):
                i.status = IntentionStatus.EXPIRED
                expired.append(i)
        return expired

    def all(self) -> list[Intention]:
        return list(self._intentions.values())

    @property
    def armed_count(self) -> int:
        return len(self.get_armed())

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for i in self._intentions.values():
            counts[i.status.value] = counts.get(i.status.value, 0) + 1
        return counts
