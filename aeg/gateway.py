"""Aeg Gateway — the only path memory takes in and out (build guide §5).

/remember screens every ingest, types it as Claim+Source+TrustSignal, records an
ImmuneEvent, and — for CLEAN memory only — cognifies the raw content into the
recall substrate with node_set facets. Quarantined memory is NEVER cognified:
innate immunity blocks a malicious ingest at the door, so it cannot appear in
recall by construction (COGNEE_NOTES: recall search is global and node_set/
dataset filtering leaks through shared-entity graph traversal, so "don't cognify"
— not filtering — is the airtight exclusion). The quarantined content is kept as
a non-indexed typed Claim for the dashboard queue and possible later release.

/recall answers from the substrate (quarantined content is absent by
construction); include_quarantined additionally surfaces the quarantine queue.
/quarantine lists quarantined claims; /scan runs the adaptive-immunity sweep;
/respond verifies quarantined memory and permanently forgets what is confirmed
bad; /reinforce strengthens a validated claim and bridges a verification note
into the permanent graph; /release reverses a quarantine; /contradictions and
/health feed the dashboard.

Run single-worker only: gateway state (per-dataset id-attribution sets + locks)
is in-process and the cognee stores are embedded/file-based.
    uv run uvicorn aeg.gateway:app --port 8080
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from aeg import antibodies, cognee_client, config, detection, health, response, screening, trust

DASHBOARD_HTML = config.REPO_ROOT / "dashboard" / "index.html"
SITE_DIR = config.REPO_ROOT / "site"
LANDING_HTML = SITE_DIR / "index.html"
HOW_HTML = SITE_DIR / "how.html"
PROOF_HTML = SITE_DIR / "proof.html"
ASSETS_DIR = SITE_DIR / "assets"
from aeg.ontology import (
    Claim,
    ClaimStatus,
    ImmuneEvent,
    Source,
    SourceKind,
    TrustSignal,
    TrustTier,
    deterministic_id,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- schemas ---------------------------------------------------------------- #

class SourceIn(BaseModel):
    kind: SourceKind = "unknown"
    identifier: str = Field(default="anonymous", max_length=200)


class RememberRequest(BaseModel):
    # max_length caps per-request LLM cost / disk (adversarial-study CRITICAL)
    content: str = Field(min_length=1, max_length=config.AEG_MAX_CONTENT_CHARS)
    source: SourceIn = SourceIn()
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    user_id: str | None = Field(default=None, max_length=64)  # Phase 6: per-user dataset


class ScreeningOut(BaseModel):
    verdict: Literal["clean", "suspect"]
    trust_tier: TrustTier
    matched_patterns: list[str]


class ClaimOut(BaseModel):
    id: str
    text: str
    subject: str
    predicate: str
    object: str
    status: ClaimStatus
    confidence: float
    data_id: str


class RememberResponse(BaseModel):
    dataset: str
    quarantined: bool  # True → content was blocked at ingest (not cognified)
    antibody: str = ""  # non-empty → a known-attack antibody pattern matched (instant block)
    data_ids: list[str]  # attributed to THIS ingest (cumulative-items diff, §2); [] if quarantined
    duplicate: bool  # clean ingest with empty diff → identical content already stored
    facets: list[str]
    screening: ScreeningOut
    source_id: str
    claims: list[ClaimOut]
    events: list[str]  # ImmuneEvent actions recorded, e.g. ["screen", "quarantine"]
    elapsed_seconds: float


class RecallRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    top_k: int = Field(default=10, ge=1, le=50)  # COGNEE_NOTES §10: >=10 for multi-hop
    lane: Literal["graph", "graph_cot"] = "graph"
    include_quarantined: bool = False  # also surface the quarantine queue
    only_context: bool = False
    user_id: str | None = Field(default=None, max_length=64)  # Phase 6: per-user dataset


class RecallEntryOut(BaseModel):
    text: str
    provenance: str | None  # cognee entry.source (graph/session/trace/...) or "quarantine"


class RecallResponseOut(BaseModel):
    query: str
    lane: str
    quarantine_excluded: bool  # True → quarantined content not surfaced (default)
    entries: list[RecallEntryOut]


class QuarantineItem(BaseModel):
    id: str
    text: str
    subject: str
    predicate: str
    object: str
    dataset: str


class QuarantineResponse(BaseModel):
    count: int
    items: list[QuarantineItem]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    datasets: list[str]
    graph: dict


class ScanRequest(BaseModel):
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    max_pairs: int = Field(default=20, ge=1, le=25)  # caps LLM verifier calls per scan
    threshold: float = Field(default=trust.VERIFIER_THRESHOLD, ge=0.0, le=1.0)


class ContradictionOut(BaseModel):
    id: str
    claim_a_id: str
    claim_b_id: str
    verdict: str
    confidence: float
    rationale: str
    detected_at: str


class ScanResponse(BaseModel):
    dataset: str
    claims_considered: int
    pairs_checked: int
    contradictions: list[ContradictionOut]
    quarantined: list[str]
    reweighted: dict[str, float]
    events: list[str]
    elapsed_seconds: float


class ReleaseRequest(BaseModel):
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    claim_id: str = Field(max_length=200)


class ReleaseResponse(BaseModel):
    dataset: str
    claim_id: str
    released: bool
    released_claims: list[str]
    restored_data_id: str
    requarantined: list[str]


class RespondRequest(BaseModel):
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    claim_id: str | None = Field(default=None, max_length=200)
    threshold: float = Field(default=trust.VERIFIER_THRESHOLD, ge=0.0, le=1.0)


class RespondResponse(BaseModel):
    dataset: str
    verified: int
    forgotten: list[str]
    forgotten_data_ids: list[str]
    released: list[str]
    skipped: list[str]
    verdicts: list[dict]
    events: list[str]
    elapsed_seconds: float


class ReinforceRequest(BaseModel):
    dataset: str = Field(default=config.DATASET_MAIN, max_length=64)
    claim_id: str = Field(max_length=200)
    note: str | None = Field(default=None, max_length=4000)


class ReinforceResponse(BaseModel):
    dataset: str
    claim_id: str
    old_confidence: float
    new_confidence: float
    session_id: str
    note: str
    bridged: bool
    requarantined: list[str]
    elapsed_seconds: float


# --- helpers ---------------------------------------------------------------- #

def attribute_new_ids(items: list[dict], seen: set[str]) -> list[str]:
    """Ids present in this RememberResult but not yet seen for the dataset.

    RememberResult.items is cumulative for the dataset's pipeline run
    (COGNEE_NOTES §2), so the delta against `seen` is what THIS call added.
    Pure — the caller updates `seen` afterward.
    """
    ids = [str(item.get("id")) for item in items if isinstance(item, dict) and item.get("id")]
    return [i for i in ids if i not in seen]


def effective_dataset(dataset: str, user_id: str | None) -> str:
    """Resolve the target dataset, applying per-user namespacing when
    AEG_MULTI_USER is on and a user_id is given (Phase 6, organizational only —
    recall stays global in embedded; real isolation needs access control)."""
    if config.AEG_MULTI_USER and user_id:
        return config.user_dataset(user_id)
    return dataset


# --- security guards (adversarial-study hardening) -------------------------- #
# The gateway is public, paid-per-LLM-call, and persistent. Every mutating/LLM
# route runs through _guard: optional shared-secret auth, a best-effort per-IP
# rate limit, and the un-spoofable global LLM budget (the wallet kill-switch).

def _reject(status: int, detail: str, headers: dict | None = None) -> None:
    raise HTTPException(status_code=status, detail=detail, headers=headers)


def _client_ip(request: Request) -> str:
    # Railway terminates TLS at a proxy; the left-most X-Forwarded-For hop is the
    # client-claimed IP. It is spoofable, so per-IP limiting is best-effort — the
    # global budget below is the real guarantee.
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def _check_api_key(request: Request, *, admin: bool) -> None:
    key = config.AEG_API_KEY
    if admin:
        # admin routes ALWAYS need the key; with none configured they are disabled
        if not key:
            _reject(403, "admin route disabled (set AEG_API_KEY to enable)")
        if request.headers.get("x-aeg-key") != key:
            _reject(401, "invalid API key")
        return
    if key and request.headers.get("x-aeg-key") != key:
        _reject(401, "missing or invalid API key")


def _rate_limit(request: Request) -> None:
    limit = config.AEG_RATE_LIMIT
    if limit <= 0:  # disabled (tests)
        return
    now = time.time()
    buckets = request.app.state.rate_buckets
    if len(buckets) > config.AEG_MAX_IP_BUCKETS:  # X-Forwarded-For is attacker-controlled
        buckets.clear()
    dq = buckets[_client_ip(request)]
    window = config.AEG_RATE_WINDOW_SECONDS
    while dq and dq[0] <= now - window:
        dq.popleft()
    if len(dq) >= limit:
        retry = int(dq[0] + window - now) + 1
        _reject(429, "rate limit exceeded", {"Retry-After": str(retry)})
    dq.append(now)


def _charge_llm(request: Request, cost: int) -> None:
    budget = config.AEG_DAILY_LLM_BUDGET
    if budget <= 0 or cost <= 0:  # disabled (tests) / non-LLM route
        return
    st = request.app.state
    now = time.time()
    if now - st.llm_window_start >= 86400:
        st.llm_window_start = now
        st.llm_calls = 0
    if st.llm_calls + cost > budget:
        _reject(503, "daily LLM budget exhausted; try again later")
    st.llm_calls += cost


def _guard(request: Request, *, llm_cost: int = 0, admin: bool = False) -> None:
    _check_api_key(request, admin=admin)
    if admin:
        return
    _rate_limit(request)
    _charge_llm(request, llm_cost)


def _valid_dataset(dataset: str) -> str:
    if not config.is_valid_dataset(dataset):
        _reject(422, "invalid dataset (must match ^aeg_[a-z0-9_]{1,48}$)")
    return dataset


def _lock_for(request: Request, dataset: str) -> asyncio.Lock:
    """Per-dataset write lock, capped so unbounded distinct datasets can't grow
    locks/seen_ids without limit (adversarial-study HIGH: dataset DoS)."""
    locks = request.app.state.locks
    if dataset not in locks and len(locks) >= config.AEG_MAX_DATASETS:
        _reject(503, "dataset capacity reached")
    return locks[dataset]


async def _cached(request: Request, key: str, ttl: float, producer):
    """Short-TTL response cache so repeated dashboard polls don't each trigger a
    full global graph export (adversarial-study MEDIUM: O(n) poll cost)."""
    cache = request.app.state.cache
    now = time.time()
    hit = cache.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = await producer()
    if len(cache) > 512:  # bound the cache (keys include client dataset)
        cache.clear()
    cache[key] = (now, value)
    return value


# --- app -------------------------------------------------------------------- #

@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Rebuild per-dataset attribution sets from the persistent store so a restart
    # doesn't mis-attribute the cumulative-items diff (COGNEE_NOTES §2;
    # adversarial-study LOW). Best-effort — never block startup.
    for ds in (config.DATASET_MAIN, config.DATASET_UNTRUSTED):
        try:
            app.state.seen_ids[ds] = set(await cognee_client.list_data_ids(ds))
        except Exception:
            pass
    yield


def create_app() -> FastAPI:
    # When AEG_API_KEY is set (self-host/prod), hide the interactive API docs so
    # the schema isn't a discovery aid for abuse (adversarial-study LOW).
    _docs = None if config.AEG_API_KEY else "/docs"
    app = FastAPI(
        title="Aeg Gateway",
        version="0.4.0",
        lifespan=_lifespan,
        docs_url=_docs,
        redoc_url=("/redoc" if _docs else None),
        openapi_url=("/openapi.json" if _docs else None),
    )
    app.state.seen_ids = defaultdict(set)  # dataset -> set[str]
    app.state.locks = defaultdict(asyncio.Lock)  # dataset -> Lock
    # recall search is GLOBAL (COGNEE_NOTES §6b), so a single global lock serializes
    # every /recall against reinforce()'s resurrection window across all datasets
    # (adversarial-study: per-dataset lock missed the cross-dataset leak).
    app.state.recall_lock = asyncio.Lock()
    app.state.rate_buckets = defaultdict(deque)  # ip -> deque[timestamps]
    app.state.llm_calls = 0
    app.state.llm_window_start = time.time()
    app.state.cache = {}  # short-TTL dashboard/health cache

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # pages are self-contained (inline styles/scripts, data:/self fonts+images);
        # same-origin fetch only. No external hosts.
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; font-src 'self' data:; connect-src 'self'; "
            "base-uri 'none'; frame-ancestors 'none'",
        )
        return resp

    @app.post("/remember", response_model=RememberResponse)
    async def remember_endpoint(req: RememberRequest, request: Request) -> RememberResponse:
        _guard(request, llm_cost=1)
        started = time.monotonic()
        req.dataset = _valid_dataset(effective_dataset(req.dataset, req.user_id))
        sc = screening.screen(req.content, req.source.kind)

        # innate memory: a replayed known attack is blocked instantly (no LLM /scan)
        hit = (await antibodies.match_antibody(req.content)
               if config.AEG_ANTIBODIES_ENABLED else None)
        blocked = sc.quarantined or hit is not None
        lock = _lock_for(request, req.dataset)  # enforce per-dataset capacity

        async def _store() -> list[str]:
            # CLEAN content is cognified into the recall substrate; QUARANTINED or
            # antibody-BLOCKED content is deliberately NOT cognified (airtight
            # exclusion, see module docstring). Serialize writes per dataset so the
            # cumulative-items diff (COGNEE_NOTES §2) is correct.
            async with lock:
                result = await cognee_client.remember(
                    req.content, dataset=req.dataset, node_set=sc.facets
                )
                seen = app.state.seen_ids[req.dataset]
                new_ids = attribute_new_ids(result.get("items", []), seen)
                seen.update(str(i.get("id")) for i in result.get("items", []) if i.get("id"))
                return new_ids

        if blocked:
            # blocked content is never cognified — do NOT spend an extraction LLM
            # call on it (adversarial-study CRITICAL: no spend on blocked content).
            # A single fallback claim keeps the quarantine queue populated.
            new_ids: list[str] = []
            drafts = [screening.ClaimDraft(subject="", predicate="", object="", text=req.content)]
        else:
            # the extraction LLM call overlaps the cognify pipeline — ~0 added wall-clock
            new_ids, drafts = await asyncio.gather(_store(), screening.extract_claims(req.content))

        source = Source(
            id=deterministic_id("source", req.source.kind, req.source.identifier),
            kind=req.source.kind,
            identifier=req.source.identifier,
            trust_tier=sc.trust_tier,
            first_seen=_now_iso(),
        )
        status: ClaimStatus = "quarantined" if blocked else "active"
        confidence = screening.TRUST_WEIGHT[sc.trust_tier]
        primary_data_id = new_ids[0] if new_ids else ""

        claims = [
            Claim(
                id=deterministic_id(req.dataset, "claim", draft.text),
                text=draft.text,
                subject=draft.subject,
                predicate=draft.predicate,
                object=draft.object,
                confidence=confidence,
                status=status,
                data_id=primary_data_id,
                dataset=req.dataset,
            )
            for draft in drafts
        ]
        signals = [
            TrustSignal(
                id=deterministic_id("trust", str(claim.id), str(source.id)),
                claim=claim,
                source=source,
                weight=confidence,
                rationale=f"{req.source.kind} source, {sc.trust_tier} tier",
            )
            for claim in claims
        ]
        events = [
            ImmuneEvent(
                action="screen",
                target=req.content[:80],
                details=f"verdict={sc.verdict} patterns={sc.matched_patterns}",
                dataset=req.dataset,
                occurred_at=_now_iso(),
            )
        ]
        if sc.quarantined:
            events.append(
                ImmuneEvent(
                    action="quarantine",
                    target=req.content[:80],
                    details=f"injection signatures: {sc.matched_patterns}",
                    dataset=req.dataset,
                    occurred_at=_now_iso(),
                )
            )
        if hit is not None:
            events.append(
                ImmuneEvent(
                    action="antibody",
                    target=req.content[:80],
                    details=f"known attack {hit.attack_type} blocked; times_seen={hit.times_seen + 1}",
                    dataset=req.dataset,
                    occurred_at=_now_iso(),
                )
            )
            await antibodies.record_antibody(
                req.dataset, pattern=hit.pattern, attack_type=hit.attack_type)

        # Persist audit events only for state-changing ingests (quarantine /
        # antibody). A routine "screen: clean" event on every ingest would grow the
        # overlay without bound (adversarial-study MEDIUM) — the overlay clears only
        # on reset_all (COGNEE_NOTES §6b), so the audit log must track ACTIONS, not
        # traffic. The response still reports "screen" so callers see it ran.
        persist_events = events if blocked else [e for e in events if e.action != "screen"]
        await cognee_client.add_data_points([source, *claims, *signals, *persist_events])

        return RememberResponse(
            dataset=req.dataset,
            quarantined=blocked,
            antibody=hit.pattern if hit is not None else "",
            data_ids=new_ids,
            duplicate=bool(not blocked and not new_ids),
            facets=sc.facets,
            screening=ScreeningOut(
                verdict=sc.verdict,
                trust_tier=sc.trust_tier,
                matched_patterns=sc.matched_patterns,
            ),
            source_id=str(source.id),
            claims=[
                ClaimOut(
                    id=str(c.id),
                    text=c.text,
                    subject=c.subject,
                    predicate=c.predicate,
                    object=c.object,
                    status=c.status,
                    confidence=c.confidence,
                    data_id=c.data_id,
                )
                for c in claims
            ],
            events=[e.action for e in events],
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    @app.post("/recall", response_model=RecallResponseOut)
    async def recall_endpoint(req: RecallRequest, request: Request) -> RecallResponseOut:
        _guard(request, llm_cost=1)
        req.dataset = _valid_dataset(effective_dataset(req.dataset, req.user_id))
        # Quarantined content was never cognified, so it is absent from the
        # substrate by construction — no filter needed on the recall itself.
        recall_kwargs = {}
        if config.AEG_TRUTH_SUBSPACE:  # rerank toward validated truth directions
            recall_kwargs["retriever_specific_config"] = {"use_truth_weight": True}
        _lock_for(request, req.dataset)  # enforce dataset capacity (bounds state)
        # Serialize against reinforce()'s dataset-wide cognify (which transiently
        # resurrects poison into the GLOBAL substrate before its re-quarantine
        # sweep) via a single global lock — recall is global, so a per-dataset lock
        # would miss a reinforce on another dataset (adversarial-study MEDIUM).
        async with request.app.state.recall_lock:
            entries = await cognee_client.recall(
                req.query,
                datasets=[req.dataset],
                top_k=req.top_k,
                query_type=cognee_client.LANES[req.lane],
                only_context=req.only_context,
                **recall_kwargs,
            )
        out = [RecallEntryOut(text=e["text"], provenance=e["source"]) for e in entries]
        if req.include_quarantined:
            # surface the quarantine queue alongside live memory (debug/inspection)
            for claim in await cognee_client.list_typed_nodes(
                "Claim", status="quarantined", dataset=req.dataset
            ):
                out.append(RecallEntryOut(text=claim.get("text", ""), provenance="quarantine"))
        return RecallResponseOut(
            query=req.query,
            lane=req.lane,
            quarantine_excluded=not req.include_quarantined,
            entries=out,
        )

    @app.get("/quarantine", response_model=QuarantineResponse)
    async def quarantine_queue(dataset: str = config.DATASET_MAIN) -> QuarantineResponse:
        _valid_dataset(dataset)
        claims = await cognee_client.list_typed_nodes(
            "Claim", status="quarantined", dataset=dataset
        )
        items = [
            QuarantineItem(
                id=c.get("id", ""), text=c.get("text", ""), subject=c.get("subject", ""),
                predicate=c.get("predicate", ""), object=c.get("object", ""),
                dataset=c.get("dataset", ""),
            )
            for c in claims
        ]
        return QuarantineResponse(count=len(items), items=items)

    @app.post("/scan", response_model=ScanResponse)
    async def scan_endpoint(req: ScanRequest, request: Request) -> ScanResponse:
        _guard(request, llm_cost=req.max_pairs)  # worst-case one verifier call per pair
        _valid_dataset(req.dataset)
        started = time.monotonic()
        # scan forgets substrate items and flips overlay state — serialize
        # against concurrent /remember cognify on the same dataset
        async with _lock_for(request, req.dataset):
            report = await detection.scan_dataset(
                req.dataset, max_pairs=req.max_pairs, threshold=req.threshold
            )
        return ScanResponse(
            dataset=report.dataset,
            claims_considered=report.claims_considered,
            pairs_checked=report.pairs_checked,
            contradictions=[ContradictionOut(**c) for c in report.contradictions],
            quarantined=report.quarantined,
            reweighted=report.reweighted,
            events=report.events,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    @app.get("/contradictions")
    async def contradictions(dataset: str = config.DATASET_MAIN) -> dict:
        _valid_dataset(dataset)
        records = await cognee_client.list_typed_nodes("Contradiction", dataset=dataset)
        items = [
            {
                "id": r.get("id", ""),
                "claim_a_id": r.get("claim_a_id", ""),
                "claim_b_id": r.get("claim_b_id", ""),
                "verdict": r.get("verdict", ""),
                "confidence": r.get("confidence", 0.0),
                "rationale": r.get("rationale", ""),
                "detected_at": r.get("detected_at", ""),
            }
            for r in records
        ]
        return {"count": len(items), "items": items}

    @app.post("/release", response_model=ReleaseResponse)
    async def release_endpoint(req: ReleaseRequest, request: Request) -> ReleaseResponse:
        _guard(request)  # no LLM, but rate-limit + optional auth
        _valid_dataset(req.dataset)
        async with _lock_for(request, req.dataset):
            result = await detection.release_claim(req.dataset, req.claim_id)
        if not result.get("released"):
            raise HTTPException(status_code=404, detail=result.get("reason", "not found"))
        return ReleaseResponse(
            dataset=req.dataset,
            claim_id=req.claim_id,
            released=True,
            released_claims=result.get("released_claims", []),
            restored_data_id=result.get("restored_data_id", ""),
            requarantined=result.get("requarantined", []),
        )

    @app.post("/respond", response_model=RespondResponse)
    async def respond_endpoint(req: RespondRequest, request: Request) -> RespondResponse:
        # respond() makes at most MAX_VERIFICATIONS_PER_CALL verifier calls, so the
        # charge is a true upper bound (not a flat under-count)
        _guard(request, llm_cost=response.MAX_VERIFICATIONS_PER_CALL)
        _valid_dataset(req.dataset)
        started = time.monotonic()
        async with _lock_for(request, req.dataset):
            report = await response.respond(
                req.dataset, claim_id=req.claim_id, threshold=req.threshold,
                max_verifications=response.MAX_VERIFICATIONS_PER_CALL,
            )
            # forgotten data ids are deterministic UUID5-of-content: a replayed
            # identical poison reuses the SAME id, so it must be re-attributable
            seen = app.state.seen_ids[req.dataset]
            for data_id in report.forgotten_data_ids:
                seen.discard(data_id)
        return RespondResponse(
            dataset=report.dataset,
            verified=report.verified,
            forgotten=report.forgotten,
            forgotten_data_ids=report.forgotten_data_ids,
            released=report.released,
            skipped=report.skipped,
            verdicts=report.verdicts,
            events=report.events,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    @app.post("/reinforce", response_model=ReinforceResponse)
    async def reinforce_endpoint(req: ReinforceRequest, request: Request) -> ReinforceResponse:
        # remember(note) + a dataset-wide improve() cognify — several LLM calls
        _guard(request, llm_cost=5)
        _valid_dataset(req.dataset)
        started = time.monotonic()
        # hold recall_lock too: reinforce's improve() resurrects poison globally
        # until its re-quarantine sweep — no /recall may run inside that window
        async with _lock_for(request, req.dataset), request.app.state.recall_lock:
            report = await response.reinforce(req.dataset, req.claim_id, note=req.note)
            if report is None:
                raise HTTPException(status_code=404, detail="claim not found")
            # the bridge created new session-derived data items — absorb them so
            # the next /remember's cumulative-items diff attributes correctly
            app.state.seen_ids[req.dataset] = set(
                await cognee_client.list_data_ids(req.dataset)
            )
        return ReinforceResponse(
            dataset=report.dataset,
            claim_id=report.claim_id,
            old_confidence=report.old_confidence,
            new_confidence=report.new_confidence,
            session_id=report.session_id,
            note=report.note,
            bridged=report.bridged,
            requarantined=report.requarantined,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_endpoint(request: Request) -> HealthResponse:
        # cached: graph_metrics runs a superlinear whole-graph query, and /health
        # is polled — don't recompute it on every hit (adversarial-study MEDIUM)
        metrics = await _cached(request, "__health__", 2.0, cognee_client.graph_metrics)
        return HealthResponse(
            status="ok",
            datasets=[
                config.DATASET_MAIN,
                config.DATASET_UNTRUSTED,
                config.DATASET_ANTIBODIES,
            ],
            graph=metrics,
        )

    # --- dashboard (Phase 5) ------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    async def landing_page() -> HTMLResponse:
        return HTMLResponse(LANDING_HTML.read_text())  # per-request: live-editable

    @app.get("/how", response_class=HTMLResponse)
    async def how_page() -> HTMLResponse:
        return HTMLResponse(HOW_HTML.read_text())

    @app.get("/proof", response_class=HTMLResponse)
    async def proof_page() -> HTMLResponse:
        return HTMLResponse(PROOF_HTML.read_text())

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page() -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML.read_text())

    @app.get("/favicon.svg")
    async def favicon():
        return FileResponse(ASSETS_DIR / "favicon.svg", media_type="image/svg+xml")

    @app.get("/assets/{filename}")
    async def asset(filename: str):
        # brand kit: css, fonts, svg marks, og image
        path = (ASSETS_DIR / filename).resolve()
        if ASSETS_DIR.resolve() not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    @app.get("/dashboard/state")
    async def dashboard_state(request: Request, dataset: str = config.DATASET_MAIN) -> dict:
        _valid_dataset(dataset)

        async def _build() -> dict:
            # Cache the GLOBAL export under one key so distinct `dataset` params all
            # reuse a single whole-graph read (an attacker cycling the dataset param
            # can no longer force one full export per request — the partition below
            # is cheap in-process). Was: two global exports per poll, per dataset.
            nodes, _ = await _cached(request, "__graph_export__", 1.5, cognee_client.export_graph)
            snap = await cognee_client.dashboard_snapshot(dataset, nodes=nodes)
            claims, contradictions, events = snap["claims"], snap["contradictions"], snap["events"]
            score = health.compute_score(claims, contradictions)
            status_by_id = {str(c["id"]): c.get("status") for c in claims}

            def is_open(conflict: dict) -> bool:
                return (status_by_id.get(str(conflict.get("claim_a_id"))) != "forgotten"
                        and status_by_id.get(str(conflict.get("claim_b_id"))) != "forgotten")

            nodes = [
                {
                    "id": str(c["id"]),
                    "label": c.get("text", "")[:70],
                    "subject": c.get("subject", ""),
                    "status": c.get("status", "active"),
                    "confidence": c.get("confidence", 0.5),
                    "trust": "high" if float(c.get("confidence", 0.5)) >= 0.5 else "low",
                }
                for c in claims
            ]
            edges = [
                {
                    "id": str(k["id"]),
                    "source": k.get("claim_a_id", ""),
                    "target": k.get("claim_b_id", ""),
                    "kind": "contradiction",
                    "verdict": k.get("verdict", ""),
                    "open": is_open(k),
                }
                for k in contradictions
            ]
            quarantine = [
                {**c, "innate": not c.get("data_id")}
                for c in claims if c.get("status") == "quarantined"
            ]
            contradiction_out = [{**k, "open": is_open(k)} for k in contradictions]
            events_sorted = sorted(events, key=lambda e: e.get("occurred_at", ""), reverse=True)
            antibody_list = sorted(
                snap["antibodies"], key=lambda a: -int(a.get("times_seen", 1)),
            )
            return {
                "dataset": dataset,
                "health": score,
                "graph": {"nodes": nodes, "edges": edges},
                "quarantine": quarantine,
                "contradictions": contradiction_out,
                "events": events_sorted,
                "antibodies": antibody_list,
            }

        # the expensive whole-graph read is cached globally inside _build; the
        # per-dataset partition/format is cheap, so no per-dataset cache key
        return await _build()

    @app.get("/events")
    async def events_endpoint(request: Request, dataset: str = config.DATASET_MAIN) -> dict:
        _valid_dataset(dataset)
        nodes, _ = await _cached(request, "__graph_export__", 1.5, cognee_client.export_graph)
        snap = await cognee_client.dashboard_snapshot(dataset, nodes=nodes)
        items = sorted(snap["events"], key=lambda e: e.get("occurred_at", ""), reverse=True)
        return {"count": len(items), "items": items}

    @app.get("/antibodies")
    async def antibodies_endpoint() -> dict:
        items = sorted(
            await cognee_client.list_typed_nodes("Antibody"),
            key=lambda a: -int(a.get("times_seen", 1)),
        )
        return {
            "count": len(items),
            "items": [
                {"pattern": a.get("pattern", ""), "attack_type": a.get("attack_type", ""),
                 "times_seen": a.get("times_seen", 1), "last_seen": a.get("last_seen", "")}
                for a in items
            ],
        }

    @app.post("/demo/reset")
    async def demo_reset(request: Request) -> dict:
        # DESTRUCTIVE global prune (COGNEE_NOTES §6b). Admin-only: requires
        # AEG_API_KEY and is disabled entirely when none is set, so it can never be
        # an unauthenticated one-click wipe of the live store (adversarial-study
        # HIGH: unauthenticated global memory wipe).
        _guard(request, admin=True)
        await cognee_client.reset_all()
        app.state.seen_ids.clear()
        app.state.locks.clear()
        app.state.cache.clear()
        return {"status": "ok"}

    return app


app = create_app()
