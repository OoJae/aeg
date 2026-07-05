"""Antibody meta-memory — innate memory of defeated attacks (Phase 6).

When the immune response permanently forgets a confirmed-bad memory, Aeg records
an Antibody: a canonical token signature of the attack. On every future ingest,
`match_antibody` checks the incoming content against known antibodies by SUBSET
containment — an instant, LLM-free block of replayed attacks. This closes the
replay/re-infection vector detection.py deferred to Phase 6.

Two-stage match: (1) lexical token-subset catches exact/near-exact replays
(reordering, case, whitespace, filler words) instantly with no embedding; (2) when
AEG_SEMANTIC_ANTIBODIES is on, an embedding-cosine fallback (≥0.82) catches
SYNONYM-SWAP / reworded near-duplicate replays that lexical misses — e.g. "Postgres
→ PostgreSQL", "MongoDB → Mongo", or a restructured sentence (cosine ~0.99 vs the
recorded attack). It deliberately stops there: a HEAVY paraphrase and the legit
truth about the same subject sit at the same cosine (~0.72), so a threshold low
enough to catch heavy paraphrase would censor the truth. Heavy paraphrase is left
to the LLM /scan (defense in depth). All cognee access goes through cognee_client.
"""

from __future__ import annotations

import math
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


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


async def record_antibody(
    dataset: str, *, pattern: str, attack_type: AttackType, sample: str = ""
) -> Antibody:
    """UPSERT an antibody by its pattern. First sighting → times_seen=1; a repeat
    (same pattern) increments times_seen and refreshes last_seen. When `sample`
    (the attack's text) is given and semantic antibodies are on, also store an
    embedding so paraphrased replays can be caught by cosine — a superset of the
    lexical block. Upsert without a sample preserves any existing embedding."""
    existing = await cognee_client.list_typed_nodes("Antibody", pattern=pattern)
    now = _now_iso()
    embedding: list[float] = []
    if sample and config.AEG_SEMANTIC_ANTIBODIES:
        try:
            embedding = [float(x) for x in (await cognee_client.embed([sample]))[0]]
        except Exception:
            embedding = []
    if existing:
        antibody = antibody_from_payload(existing[0])  # carries prior embedding
        antibody.times_seen = int(existing[0].get("times_seen", 1)) + 1
        antibody.last_seen = now
        if embedding:
            antibody.embedding, antibody.sample = embedding, sample
    else:
        antibody = Antibody(
            id=deterministic_id("antibody", pattern),
            pattern=pattern,
            attack_type=attack_type,
            times_seen=1,
            last_seen=now,
            dataset=config.DATASET_ANTIBODIES,
            embedding=embedding,
            sample=sample,
        )
    await cognee_client.add_data_points([antibody])
    return antibody


async def match_antibody(content: str) -> Antibody | None:
    """Instant, LLM-free lookup for a known attack. Global — an attack learned
    anywhere is remembered everywhere. Two stages:

    1. LEXICAL — the most specific antibody whose core token-set is contained in
       the incoming content (exact/near-exact replays; no embedding needed).
    2. SEMANTIC — if lexical misses and AEG_SEMANTIC_ANTIBODIES is on, cosine of
       the content's embedding vs stored antibody embeddings ≥ threshold catches
       PARAPHRASED replays that share no distinctive tokens. Only runs when some
       antibody actually carries an embedding, so the lexical/free path is unchanged.
    """
    content_tokens = tokens(content)
    known = await cognee_client.list_typed_nodes("Antibody")
    if len(known) > MAX_ANTIBODIES_SCANNED:  # check the most-seen first, bounded work
        known = sorted(known, key=lambda a: -int(a.get("times_seen", 1)))[:MAX_ANTIBODIES_SCANNED]

    best, best_len = None, -1
    for payload in known:
        core = frozenset(payload.get("pattern", "").split("|")) - {""}
        if len(core) >= MIN_CORE_TOKENS and core <= content_tokens and len(core) > best_len:
            best, best_len = payload, len(core)
    if best is not None:
        return antibody_from_payload(best)

    if config.AEG_SEMANTIC_ANTIBODIES:
        embedded = [p for p in known if p.get("embedding")]
        if embedded:
            try:
                query = [float(x) for x in (await cognee_client.embed([content]))[0]]
            except Exception:
                query = []
            if query:
                best_sim, best_p = config.AEG_ANTIBODY_SIM_THRESHOLD, None
                for p in embedded:
                    sim = _cosine(query, p.get("embedding") or [])
                    if sim >= best_sim:
                        best_sim, best_p = sim, p
                if best_p is not None:
                    return antibody_from_payload(best_p)
    return None
