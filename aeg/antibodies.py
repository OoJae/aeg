"""Antibody meta-memory — innate memory of defeated attacks (Phase 6).

When the immune response permanently forgets a confirmed-bad memory, Aeg records
an Antibody: a canonical token signature of the attack. On every future ingest,
`match_antibody` checks the incoming content against known antibodies by SUBSET
containment — an instant, LLM-free block of replayed attacks. This closes the
replay/re-infection vector detection.py deferred to Phase 6.

Fingerprint scope (honest ceiling): catches exact and near-exact replays
(reordering, case, whitespace, filler-word changes). Synonym-level paraphrase
("Mongo" vs "MongoDB") is NOT caught here — that still falls to the LLM /scan
(defense in depth). All cognee access goes through cognee_client.
"""

from __future__ import annotations

from datetime import datetime, timezone

from aeg import cognee_client, config
from aeg.detection import tokens  # reuse the exact tokenizer + STOPWORDS
from aeg.ontology import Antibody, AttackType, antibody_from_payload, deterministic_id

# An antibody core below this is too generic and would block benign supersets
# that merely happen to contain the tokens (adversarial-study MEDIUM: antibody
# false-positive censorship). 4 raises specificity without missing real replays.
MIN_CORE_TOKENS = 4
# Bound the per-ingest scan so a large antibody set can't make every /remember
# superlinear; most-seen antibodies are checked first.
MAX_ANTIBODIES_SCANNED = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _field(obj, name: str) -> str:
    if isinstance(obj, dict):
        return obj.get(name, "") or ""
    return getattr(obj, name, "") or ""


def fingerprint(source) -> str:
    """Canonical sorted token signature. Accepts raw content (str) or a claim
    (dict/DataPoint). Numbers are kept — they are the anti-collision key that
    separates a poisoned value ("9999") from the truth ("100")."""
    if isinstance(source, str):
        core = tokens(source)
    else:
        text = _field(source, "text")
        core = tokens(text) | tokens(_field(source, "subject")) | tokens(_field(source, "object"))
        core = core or tokens(text)
    return "|".join(sorted(core))


async def record_antibody(dataset: str, *, pattern: str, attack_type: AttackType) -> Antibody:
    """UPSERT an antibody by its pattern. First sighting → times_seen=1; a repeat
    (same pattern) increments times_seen and refreshes last_seen."""
    existing = await cognee_client.list_typed_nodes("Antibody", pattern=pattern)
    now = _now_iso()
    if existing:
        antibody = antibody_from_payload(existing[0])
        antibody.times_seen = int(existing[0].get("times_seen", 1)) + 1
        antibody.last_seen = now
    else:
        antibody = Antibody(
            id=deterministic_id("antibody", pattern),
            pattern=pattern,
            attack_type=attack_type,
            times_seen=1,
            last_seen=now,
            dataset=config.DATASET_ANTIBODIES,
        )
    await cognee_client.add_data_points([antibody])
    return antibody


async def match_antibody(content: str) -> Antibody | None:
    """Instant, LLM-free lookup: return the most specific known antibody whose
    core token-set is contained in the incoming content, or None. Global — an
    attack learned anywhere is remembered everywhere (correct immune semantics,
    and recall is global anyway)."""
    content_tokens = tokens(content)
    best: dict | None = None
    best_len = -1
    known = await cognee_client.list_typed_nodes("Antibody")
    if len(known) > MAX_ANTIBODIES_SCANNED:  # check the most-seen first, bounded work
        known = sorted(known, key=lambda a: -int(a.get("times_seen", 1)))[:MAX_ANTIBODIES_SCANNED]
    for payload in known:
        core = frozenset(payload.get("pattern", "").split("|")) - {""}
        if len(core) >= MIN_CORE_TOKENS and core <= content_tokens and len(core) > best_len:
            best, best_len = payload, len(core)
    return antibody_from_payload(best) if best else None
