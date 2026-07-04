# COGNEE_NOTES — the verified API contract for Aeg

Every statement below was **empirically confirmed against the installed package**
by `scripts/verify_cognee_api.py` (18/18 checks PASS; raw evidence in
`docs/verify_output.txt`, regenerable with
`uv run scripts/verify_cognee_api.py --include-destructive`). Each section cites
the check IDs that back it. `aeg/cognee_client.py` is built against THIS file,
not against docs or memory.

Binding environment (P0.1, P0.4):

| | |
|---|---|
| cognee | **1.2.2** (PyPI, pinned exactly in pyproject.toml) |
| Python | 3.12.13 (uv-managed; cognee supports >=3.10,<3.15) |
| Relational | SQLite (`db_provider: sqlite`, file-based) |
| Vector | LanceDB (file-based) |
| Graph | Ladybug (`graph_database_provider: ladybug`; "kuzu" is a legacy alias) |
| LLM | `custom` provider → MiMo v2.5 Pro via OpenAI-compatible endpoint (litellm + instructor `json_mode`) |
| Embeddings | `fastembed` local, `sentence-transformers/all-MiniLM-L6-v2`, 384 dims (no API key) |

All stores are file-based with zero external services. State isolation is done by
setting `DATA_ROOT_DIRECTORY` / `SYSTEM_ROOT_DIRECTORY` / `CACHE_ROOT_DIRECTORY`
**before** `import cognee` — without them, cognee writes inside site-packages.

## 1. Top-level surface & canonicity (P0.1, P0.3)

The canonical v1 surface is the async quartet, re-exported at top level:
`cognee.remember()`, `cognee.recall()`, `cognee.improve()`, `cognee.forget()`.
The legacy layer (`add`/`cognify`/`search`/`memify`/`delete`) remains exported and
functional; `remember` wraps add+cognify, `improve` wraps memify (§3). All are
`async` — every call is awaited.

**Param asymmetry is real and must not be "normalized" incorrectly** (P0.3):
- `remember(data, dataset_name="main_dataset", ...)`
- `improve(dataset="main_dataset", ...)` / `forget(dataset=..., ...)`
- `recall(query_text, datasets=[...], ...)`

Key signatures (verbatim in Appendix A / P0.3 evidence):

- `remember(data, dataset_name='main_dataset', *, session_id=None, chunk_size=None, chunker=None, custom_prompt=None, run_in_background=False, self_improvement=True, session_ids=None, **RememberKwargs) -> RememberResult`
  - `RememberKwargs` (TypedDict — **invisible to `inspect.signature`**, probe by calling): `graph_model, node_set, dataset_id, preferred_loaders, incremental_loading, data_per_batch, chunks_per_batch, user, vector_db_config, graph_db_config, content_type, skill_improvement, skills_text, skill_name, primary_key, write_disposition, query, max_rows_per_table, llm_config, embedding_config`
- `recall(query_text, query_type=None, *, datasets=None, dataset_ids=None, top_k=15, auto_route=True, scope=None, node_name=None, node_name_filter_operator='OR', only_context=False, session_id=None, feedback_influence=0.0, verbose=False, retriever_specific_config=None, neighborhood_depth=None, include_references=False, ...) -> list[RecallResponse]`
- `improve(dataset='main_dataset', *, run_in_background=False, node_name=None, session_ids=None, build_global_context_index=False, build_truth_subspace=False, **ImproveKwargs)`
  - `ImproveKwargs`: `extraction_tasks, enrichment_tasks, data, node_type, user, vector_db_config, graph_db_config, feedback_alpha`
- `forget(*, data_id=None, dataset=None, dataset_id=None, everything=False, memory_only=False, user=None) -> dict`
- `SearchType` has 17 members; default lane is `GRAPH_COMPLETION`.

## 2. Result shapes & data-id capture (P1.1, P1.2)

`RememberResult.to_dict()` keys: `status, dataset_name, dataset_id,
pipeline_run_id, items_processed, elapsed_seconds, items`.

**data_id capture (the forget() enabler):** `items` is a list of `{"id": ...}`
dicts. ⚠️ It is **cumulative for the dataset's pipeline run** — after a second
`remember()` into the same dataset it contained BOTH items' ids. To attribute
"the id of what I just wrote", diff against the previous item set (or use the
fallback: `datasets.list_datasets()` → match name → `datasets.list_data(dataset_id=...)`).
Ids are deterministic (UUID5 of content hash): identical text → identical data_id,
even across datasets.

`recall()` returns typed Pydantic entries discriminated by a `source` attribute:
`'graph' | 'graph_context' | 'session' | 'session_context' | 'trace'`
(P1.1 observed `ResponseGraphEntry` with `source='graph'`). `only_context=True`
returns retrieved context without a synthesized LLM answer (P2.5).
⚠️ `recall()` on a dataset that was fully forgotten **raises
`DatasetNotFoundError`** (from `cognee.modules.data.exceptions.exceptions`) —
the wrapper catches this and returns empty (P1.2).

## 3. improve() ↔ memify contract (P2.1, P1.6)

- `improve()` **delegates to `memify()`** (confirmed via `inspect.getsource`);
  `memify` stays exported for lower-level control.
- Without `session_ids`: enrichment only — fast (0.1 s on a small graph).
- Returns `dict[dataset_uuid -> PipelineRunCompleted]` (status, pipeline_run_id,
  dataset_id/name, data_ingestion_info).
- `feedback_alpha=0.8` accepted via kwargs (probe-by-calling; not in the visible
  signature).
- ⚠️ **Phase-4 addendum — the resurrection vector (source-confirmed):**
  `improve(session_ids=[...])` runs `cognify_session`, which calls
  `cognee.add(...)` + `cognee.cognify(datasets=[dataset_id])` — a dataset-WIDE
  incremental cognify that re-processes ANY item whose cognify status was reset,
  i.e. it RESURRECTS still-memory_only-quarantined items. Enrichment-only
  `improve()` (no session_ids) never touches Data records and is safe. Aeg's
  `response.reinforce()` therefore ends with a resurrection sweep (re-forget
  memory_only of everything still quarantined), and respond() runs before
  reinforce() by convention.
- **Full forget after memory_only quarantine works** (Phase-4 gate test): the
  ownership-ledger rows are already gone, the legacy soft-delete path no-ops on
  the empty subgraph, and the raw Data record is removed — `status: success`,
  id gone from `list_data`.

## 4. forget() granularity — VERIFIED + strategy (P1.2, P2.5, PX.1)

The whole immune-response design rests on this; all confirmed on 1.2.2:

- **Item-level**: `forget(data_id=<uuid>, dataset=<name>)` surgically removed
  fact A ("Atlas uses Postgres") while fact B ("Maya leads Atlas") stayed
  recallable, **and the shared "Project Atlas" entity node survived** (ownership
  metadata per source document). Returns `{'data_id': ..., 'dataset_id': ...,
  'status': 'success'}` in ~0.1–0.2 s with **zero LLM calls**.
- **Dataset-level**: `forget(dataset=<name>)` wipes the dataset; subsequent
  recall raises `DatasetNotFoundError`.
- **memory_only=True**: graph+vectors wiped, raw data records kept
  (`list_data` still returns the item), and `cognify(datasets=[...])` fully
  restores recall. A reversible "hard quarantine" primitive.
- **Phase-3 addendum (P3.A, runtime-confirmed + source-read):** memory_only works
  at ITEM level too — `forget(data_id=X, dataset=ds, memory_only=True)` removes
  only that item's graph+vectors (sibling facts stay recallable; shared nodes
  co-owned by other items are preserved), keeps the Data record, and resets only
  that item's cognify pipeline status. `cognify(datasets=[ds])` (internally
  `use_pipeline_cache=False` + `incremental_loading=True`) then re-processes
  ONLY the reset item(s) — 7.2s for one item — and the restored content keeps the
  SAME data_id. This is Aeg's detection-time reversible quarantine. Caveat:
  release is dataset-granular (restores ALL reset items) — re-forget the ones
  that must stay quarantined afterward (~0.2s each, zero LLM).
- **everything=True**: global wipe; 0 datasets remain (PX.1; demo reset only).
- NOT supported: node-level, time-range, or query-based deletion.

**AGREED STRATEGY** (Checkpoint 1): primary = item-level `forget(data_id, dataset)`
with data_ids captured at ingest (§2). Pre-wired fallback = segregate untrusted
ingests into a droppable `aeg_untrusted` dataset + delete-and-reingest-survivors.
Soft quarantine = `node_set` tag exclusion (§6); reversible before any forget.

## 5. Custom DataPoints (P1.5)

- Both import paths work: `from cognee.low_level import DataPoint` and
  `from cognee.infrastructure.engine import DataPoint, Embeddable, Dedup`.
- Subclass + `metadata: dict = {"index_fields": [...]}` — no registration needed.
  Inherited fields include `id, created_at, updated_at, version, type,
  belongs_to_set, feedback_weight, importance_weight, source_content_hash`.
- Direct instance insert: `from cognee.tasks.storage import add_data_points;
  await add_data_points([claim_a, claim_b])` → typed nodes appear in the graph.
- ⚠️ **No dedup observed** for identical re-inserts — neither bare instances nor
  `Annotated[str, Dedup()]` collapsed duplicates. For stable Claim identity,
  set deterministic `id=uuid5(...)` explicitly at construction.
- **Typed extraction works**: `remember(text, graph_model=MyRootType)` produced
  typed nodes (`VfyPerson`×2, `VfyProject`×4) from free text. Use a root
  "container" DataPoint with list-fields of the child types.
- **Phase-3 addendum (P3.B, runtime-confirmed + source-read):**
  - `add_data_points` with an EXISTING node id is an **UPSERT** (ladybug
    `add_nodes` = `MERGE (n {id}) ... ON MATCH SET n.properties = ...`) — the
    status/confidence-flip mechanism. ⚠️ MERGE replaces properties WHOLESALE:
    an upsert must carry ALL fields (see `ontology.claim_from_payload`), and a
    batch must never embed STALE copies of a DataPoint it also updates — build
    one updated instance and reference it everywhere in that batch.
  - Relationship-typed DataPoint fields (e.g. `Contradiction.claim_a`) become
    graph **EDGES**, not payload properties — readable references need scalar
    mirror fields (`claim_a_id`).
  - `get_graph_data()` node payloads exclude the node's own id (popped at write
    time) — `cognee_client.list_typed_nodes` injects it as `payload["id"]`.

## 6. node_set tagging & quarantine filtering (P1.3, P1.4)

- Ingest: `remember(..., node_set=["source:tool", "quarantine:true"])` (via
  RememberKwargs; `add()` has it as a named param).
- **Filtering works in-process**: with two contradictory facts in one dataset,
  `recall(..., node_name=["quarantine:false"])` returned ONLY the clean fact;
  `node_name=["quarantine:true"]` returned only the poisoned one.
  `node_name_filter_operator="AND"` works.
- **Propagation is full-depth** (P1.4): `belongs_to_set` edges reach
  `Entity`, `TextSummary`, `DocumentChunk`, AND `TextDocument` nodes — quarantine
  tags cover LLM-derived knowledge, not just raw chunks.
- `scope=` is a **lane selector** (`graph`/`session`/`trace`...), NOT a tag
  filter (confirmed: tags leaked through scope).
- CHUNKS-lane + node_name appeared filtered in our run, but docs say tag
  filtering is only supported on GRAPH_COMPLETION-family lanes — do not rely on
  it for vector lanes.
- ⚠️ **Phase-2 addendum — node_set filtering is NOT airtight under a busy graph.**
  P1.3's clean 2-fact result did not hold once the graph was populated with more
  nodes (typed overlay + other datasets): `GRAPH_COMPLETION` traversal reaches
  quarantine:true content by hopping through untagged bridge nodes from a
  quarantine:false seed. Filtering is a *soft* signal, good for dashboard
  drill-down; it is **not** the recall-exclusion boundary. See §6b.

## 6b. Recall scoping & the airtight-exclusion finding (Phase 2)

Empirically established while building the gateway:

- **`recall(datasets=[X])` does NOT isolate the search** in the embedded,
  `ENABLE_BACKEND_ACCESS_CONTROL=false` / `REQUIRE_AUTHENTICATION=false` config:
  content ingested only into dataset B (disjoint entities) was returned by
  `recall(datasets=["A"])`. Search is effectively global (single default user).
  The `datasets` param scopes writes and dataset lifecycle (forget), not read.
- **The only airtight ways to keep content out of recall are (a) never cognify
  it, or (b) `forget()` it (deletion, P1.2).** Filtering (node_set) and scoping
  (datasets) both leak via shared-entity graph traversal.
- **Consequence for Aeg's quarantine:** ingest-time quarantine = *do not cognify*
  the content (innate immunity blocks it at the door; it cannot appear in recall
  by construction). The raw text is kept as a non-indexed typed `Claim`
  (status="quarantined") for the dashboard queue and possible later release.
  Detection-time quarantine (Phase 3) will use `forget()` on already-cognified
  poison.
- ⚠️ **`add_data_points` overlay nodes are NOT removed by `forget()` /
  `forget_everything()`** — they have no dataset ownership (`source_pipeline:
  None`, `belongs_to_set: None`). The typed audit overlay accumulates for the
  process lifetime; scope reads by an explicit field (e.g. `dataset=...`) via
  `cognee_client.list_typed_nodes`. Because the overlay is **non-indexed**
  (`index_fields: []`), it never seeds recall, so accumulation does not pollute
  answers — but tests reset via a fresh scratch dir, not `forget_everything()`.
- **Phase-5 addendum — clearing the overlay needs `prune`, not `forget`.**
  `cognee.prune.prune_data()` + `cognee.prune.prune_system(graph=True,
  vector=True, metadata=True, cache=True)` (signature: `prune_system(graph=True,
  vector=True, metadata=False, cache=True)`) is the only thing that wipes the
  accumulating typed overlay — wrapped as `cognee_client.reset_all()` for the
  dashboard's `/demo/reset`. It is GLOBAL (all datasets); the authoritative
  clean slate for the live demo is restarting `demo/serve_dashboard.py`, which
  rmtrees its scratch dir before importing cognee. `ImmuneEvent` gained a
  `dataset` scalar this phase so the threat feed is dataset-scopable via
  `list_typed_nodes`/`overlay_snapshot`.

## 7. Session memory & bridging (P1.6, P2.4)

- `remember(fact, dataset_name=ds, session_id=s, self_improvement=False)` writes
  to the fs cache in **0.07 s** (no chunking/LLM/embedding).
- Session facts are visible to `recall(..., session_id=s)` and **invisible** to
  permanent recall — until `improve(dataset=ds, session_ids=[s])` bridges them
  into the graph (grouped under the `user_sessions_from_cache` node set,
  confirmed present).
- ⚠️ `self_improvement` defaults to **True** on remember() and spawns a
  background improve — Aeg defaults it OFF for determinism and bridges
  explicitly.
- Feedback: `cognee.session.add_feedback(session_id, qa_id, feedback_text=None,
  feedback_score=None) -> bool` accepted score=1; `improve(feedback_alpha=0.8)`
  ran; weight readback surface exists on the graph engine.

## 8. Graph export for the dashboard (P2.2)

- `from cognee.infrastructure.databases.graph import get_graph_engine;
  nodes, edges = await (await get_graph_engine()).get_graph_data()` — **global**
  (not dataset-scoped) nodes/edges lists.
- `await engine.get_graph_metrics()` → `num_nodes, num_edges, mean_degree,
  edge_density, num_connected_components, sizes_of_connected_components, ...`
  (health-score raw material).
- `await cognee.visualize_graph(path)` writes a self-contained interactive HTML.
- `await cognee.export(dataset=<name>)` → `GraphSnapshot(dataset, nodes, edges)`
  — the **dataset-scoped** snapshot for before/after views.

## 9. Truth-subspace reranking (P3.2) — stretch flag

Surface present on 1.2.2: `improve(build_truth_subspace=True)` and
`recall(retriever_specific_config={"use_truth_weight": True})`. Availability
verified only; building/using it is Phase 6 stretch.

## 10. Graph-traversal vs vector provenance (P1.1, P2.3)

- The `query_type` selects the lane: `CHUNKS` = pure vector similarity (returns
  raw chunks, no synthesized answer), `GRAPH_COMPLETION` family = graph
  traversal. Each recall entry's `source` attribute tags provenance.
- 2-hop proof: "What database does the project led by Maya Chen use?" was
  answered ("Postgres") **only by the graph lane**; CHUNKS returned raw chunks.
- ⚠️ **Retrieval variance**: with 384-dim local embeddings + nondeterministic
  extraction, plain GRAPH_COMPLETION sometimes missed the bridging triplet;
  `GRAPH_COMPLETION_COT` (and higher top_k) landed it. Demo guidance: ingest
  **one fact per remember() call** (list-form ingestion produced weaker graphs),
  use `top_k>=10`, prefer `GRAPH_COMPLETION_COT` for multi-hop showcases.

## 11. Config & provider matrix (P3.1, P0.4)

- Active: `LLM_PROVIDER=custom`, `LLM_MODEL=openai/mimo-v2.5-pro`,
  `LLM_ENDPOINT=<OpenAI-compatible URL>`, `LLM_API_KEY=...` → GenericAPIAdapter
  (litellm, instructor **json_mode** — no function-calling requirement).
- Embeddings: `EMBEDDING_PROVIDER=fastembed` (local, keyless). ⚠️ Without
  `EMBEDDING_*` set, cognee sends `LLM_API_KEY` to OpenAI embeddings — that trap
  breaks any non-OpenAI-only setup.
- Cloud switch exists: `cognee.serve(url=..., api_key=...) -> CloudClient`;
  Postgres via config (later-phase profile flags in `aeg/config.py`).
- `setup()` is NOT required before first use on 1.2.2 embedded (`remember()`
  self-initializes; probed cold). Top-level `cognee.setup` does not exist;
  `cognee.modules.engine.operations.setup.setup` does.

## 12. Wrapper contract for `aeg/cognee_client.py`

One function per cognee op, 1:1 names, built on the sections above:

| wrapper | delegates to | contract notes |
|---|---|---|
| `remember()` | `cognee.remember(data, dataset_name=...)` | `self_improvement=False` default (§7); returns dict incl. `items` ids (§2); `node_set`/`graph_model` pass through kwargs (§1) |
| `recall()` | `cognee.recall(query, datasets=[...])` | catches `DatasetNotFoundError` → `[]` (§2); normalizes entries to `{text, source, raw}` |
| `improve()` | `cognee.improve(dataset=...)` | `feedback_alpha` forwarded only when set (§3) |
| `forget()` | `cognee.forget(data_id=..., dataset=...)` | item-level primary (§4); `everything=True` NOT exposed here — separate guarded `forget_everything()` |
| `export_graph()/graph_metrics()/export_snapshot()` | graph engine + `cognee.export` | §8 |
| `recall_diff()` | recall → action → recall | the Phase-4 assertion primitive |

## 13. Uncertainties / flagged unknowns

- `Dedup()` annotation had no observable effect (§5) — deterministic ids must be
  explicit. Not re-tested against other cognee versions.
- CHUNKS-lane tag filtering worked once but is undocumented — treat as
  unsupported (§6).
- Multi-hop retrieval quality depends on extraction nondeterminism + embedding
  model (§10); MiMo + MiniLM is the floor, not the ceiling.
- `get_graph_data()` is global; dataset-scoped views must use
  `cognee.export(dataset=...)` or filter client-side (§8).
- Anthropic-as-LLM path untested here (config-level support exists; embeddings
  would also need fastembed/OpenAI).

## Appendix A — verify run RESULT lines (18/18)

```
RESULT|P0.1|PASS|cognee 1.2.2 exposes remember/recall/improve/forget
RESULT|P0.2|PASS|prune_data + prune_system(metadata=True, cache=True) reset the isolated stores
RESULT|P0.3|PASS|signatures recorded verbatim; RememberKwargs/ImproveKwargs dumped (invisible to inspect)
RESULT|P0.4|PASS|embedded store config + setup() locations recorded
RESULT|P1.1|PASS|fact written via remember() and recovered via recall(); shapes recorded
RESULT|P1.2|PASS|forget(data_id=...) removed fact A, kept fact B, shared 'Atlas' entity survived; dataset-level forget wiped the rest; maya_after_wipe=False
RESULT|P1.3|PASS|quarantine:false filter excluded the poisoned fact and kept the clean one
RESULT|P1.4|PASS|belongs_to_set edges reach extracted Entity nodes — quarantine tags cover derived knowledge
RESULT|P1.5|PASS|direct insert OK (2 typed nodes); Annotated[str, Dedup()] -> 2 node(s) after 2 identical inserts; remember(graph_model=...) typed extraction: True
RESULT|P1.6|PASS|session write near-instant (0.07s), isolated from permanent recall, and bridged into the graph by improve(session_ids=[...])
RESULT|P2.1|PASS|improve() runs on a cognified dataset, accepts feedback_alpha via kwargs, and delegates to memify()
RESULT|P2.2|PASS|dashboard can use get_graph_data (53n/64e) + metrics + visualize_graph HTML
RESULT|P2.3|PASS|2-hop answer emerges from the graph lane (GRAPH_COMPLETION_COT); CHUNKS lane returns only raw chunks
RESULT|P2.4|PASS|feedback attach + improve(feedback_alpha) ran; weight readback surface recorded
RESULT|P2.5|PASS|only_context path returns raw context; forget() completed in 0.2s
RESULT|P3.1|PASS|provider matrix recorded (record-only, never fails)
RESULT|P3.2|PASS|truth-subspace surface present (improve flag + retriever_specific_config)
RESULT|PX.1|PASS|global wipe ran; 0 datasets remain
```

Full evidence (signatures, dumps, timings): `docs/verify_output.txt`.

## Phase-6 addendum — stretch items (probed, gated, honest scopes)

Each stretch item is behind a flag (`config.AEG_*`), default-safe so the embedded
spine (`run_demo.py` + 56 base tests) is never at risk.

- **Antibody meta-memory** (`AEG_ANTIBODIES_ENABLED`, default **true**): defeated
  attacks are recorded as global-overlay `Antibody` nodes (token-set `pattern`,
  numbers kept as the anti-collision key); a replayed attack is blocked at ingest
  by SUBSET containment (`core ⊆ tokens(content)`, MIN_CORE_TOKENS=3) — instant,
  no LLM. Catches exact/near-exact replays (reorder/case/whitespace/filler), NOT
  synonym paraphrase (that still falls to the LLM `/scan` — defense in depth).
  Antibodies survive `forget` (overlay), cleared only by `reset_all`.
- **Truth-subspace reranking** (`AEG_TRUTH_SUBSPACE`, default **false**): probe
  `scripts/verify_phase6_truth.py` — both `improve(build_truth_subspace=True)` and
  `recall(retriever_specific_config={"use_truth_weight": True})` RUN cleanly on
  MiMo+fastembed (P6.TS-build/recall PASS). Wired behind the flag; kept default-off
  because cognee docs mark it experimental/unvalidated and it has no visible demo
  payoff. `use_truth_weight` is a KEY inside `retriever_specific_config`, not a
  bare kwarg.
- **Multi-user isolation** (`AEG_MULTI_USER`, default **false**): probe
  `scripts/verify_phase6_multiuser.py` — enabling `ENABLE_BACKEND_ACCESS_CONTROL`
  boots and the round-trip survives, but REAL per-user recall isolation needs
  cognee user objects / authenticated context (heavier auth, spine risk). Ships as
  **organizational namespacing only**: `user_id` → `aeg_user_<id>` dataset via
  `config.user_dataset`. Honest limitation: recall stays global in the embedded
  access-control-off config (§6b) — true cross-tenant enforcement needs access
  control or the Postgres/Cloud profile.
- **Postgres profile** (`AEG_PROFILE=postgres`, default `embedded`): relational +
  vector move to Postgres/pgvector (env vars `DB_PROVIDER/DB_HOST/DB_PORT/DB_NAME/
  DB_USERNAME/DB_PASSWORD`, `VECTOR_DB_PROVIDER=pgvector`); graph stays embedded
  Ladybug (no second external service). `docker-compose.yml` (pgvector/pg16) +
  the `postgres` optional extra (uses `cognee[postgres-binary]` → psycopg2-binary,
  no native build). Config-resolve unit-tested without Docker
  (`test_phase6.py::test_postgres_profile_resolves_env`, cognee inits lazily §11)
  AND **demo-verified live**: `docker compose up -d` + `AEG_PROFILE=postgres
  uv run python demo/run_demo.py` runs the full 8-step poison→heal loop with
  relational + vector on Postgres/pgvector (confirmed: `postgresql+asyncpg://`
  connection, `vector` extension enabled, `data`/`datasets`/`Entity_name`/
  `*Vector_text` tables in Postgres) while the graph stays embedded Ladybug.
