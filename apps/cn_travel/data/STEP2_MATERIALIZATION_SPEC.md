# Step 2: Grounded Scenario Materialization

This document is the acceptance contract for
`data/scripts/2_materialize.py`.

## 1. Objective

Convert every immutable Step 1 storyboard row into a concrete, grounded scenario
package. Step 3 renders natural conversation text entirely from the captured
facts and tool evidence.

```text
1_storyboard.json row
        ↓
canonical context, slots, turn intentions and response policy
        ↓
validated tool plan
        ↓
validated synthetic tool evidence, with cacheable executions captured once
        ↓
2_materialized/{idx:04d}.json
```

Step 2 and Step 3 have separate ownership:

- Step 2 chooses and validates facts, entities, dates, tool arguments, tool results and route semantics.
- Step 3 writes varied user and assistant prose around those frozen facts.

## 2. Ownership and boundaries

1. `data/1_storyboard.json` and `data/scripts/1_storyboard.py` remain
   byte-identical inputs.
2. Each Step 1 row is an immutable label.
3. Step 2 emits structured scenarios; Step 3 owns OpenAI conversation prose.
4. Deterministic program logic selects dates, entities, tools, arguments,
   statuses, and route branches.
5. Step 3 reuses the exact tool evidence captured by Step 2.
6. `expect`, `route`, `edges`, `turns`, `slots`, and `omit` retain their Step 1
   values.
7. Rows that cannot satisfy their Step 1 label produce structured attempt
   events; a complete seal requires every selected row to pass.
8. Step 2 artifacts reside under `data/2_materialized/`.
9. Generation clients reside under `data/clients/`; Step 2 consumes runtime
   contracts from `src/` through their public interfaces.
10. Step 2 tests reside under `data/tests/`.

## 3. Pipeline position and target paths

| Item | Path |
|---|---|
| Input | `data/1_storyboard.json` |
| Step 2 specification | `data/STEP2_MATERIALIZATION_SPEC.md` |
| Target implementation | `data/scripts/2_materialize.py` |
| Accepted records | `data/2_materialized/{idx:04d}.json` |
| Run manifest | `data/2_materialized/manifest.json` |
| Attempt ledger | `data/2_materialized/attempts.jsonl` |
| Generation clients | `data/clients/` |
| Source guide corpus | `data/knowledge_base/travel_guides/` |
| Step 3 target input | the sealed Step 2 manifest and its accepted records |
| Step 3 target output | `data/3_conversations/` |

## 4. Sources of truth

The materializer reads and records the following source identities at run start:

- [`1_storyboard.json`](1_storyboard.json)
- [`city_registry.json`](city_registry.json)
- [`city_weather_id.json`](city_weather_id.json)
- [`china_cities_list_out_of_scope.json`](china_cities_list_out_of_scope.json)
- the workflow route tables in [`README_CN TRAVEL.md`](../README_CN%20TRAVEL.md)
- [`../src/cn_travel/business_logic/system_prompt.md`](../src/cn_travel/business_logic/system_prompt.md)
- [`../src/cn_travel/business_logic/user_info.md`](../src/cn_travel/business_logic/user_info.md)
- [`../src/cn_travel/business_logic/tool_schemas.json`](../src/cn_travel/business_logic/tool_schemas.json)
- [`../src/cn_travel/business_logic/contracts.py`](../src/cn_travel/business_logic/contracts.py)
- [`../src/cn_travel/service/result.py`](../src/cn_travel/service/result.py)
- `get_*.py` tool modules and active underscore-prefixed provider modules under
  `../src/cn_travel/tool/`
- the materializer source identities, deterministic configuration, eligible
  profile-pool size, seed, logical date, system date and forecast horizon

The default run fingerprint is the SHA-256 of canonical sorted JSON containing the
registry, weather map, boundary, governing route document, business contracts,
tool/provider identities, profile-pool size, seed, dates, forecast horizon and
configuration. `storyboard_sha256` and `materializer_sha256` remain explicit
archive identities in `manifest.sources`; the fingerprint excludes those two
fields. With `--adopt-fingerprint`, `manifest.fingerprint` retains the existing
artifact-lineage identifier while `manifest.sources` records the active source
identities and accumulates materializer hashes. `route_manifest()` in the
materializer supplies planning and validation behavior.

`manifest.sources` uses these keys:

- `registry_sha256`, `weather_map_sha256`, `boundary_sha256` and
  `routes_doc_sha256`;
- `system_prompt_sha256`, `user_info_sha256`, `tool_schema_sha256`,
  `contract_sha256` and `spec_result_sha256`;
- `tools_sha256` and `backends_sha256`, each a filename-to-SHA-256 map;
- `profile_pool_size`, `seed`, `logical_today`, `system_today`,
  `forecast_available_through` and `config`;
- `storyboard_sha256` and the list `materializer_sha256`.

## 5. Input contract

`data/1_storyboard.json` is an array. Array position is the stable `idx` for the
declared storyboard version. A typical row is:

```json
{
  "workflow": 1,
  "route": "W1-A",
  "edges": ["e3", "e6", "e8", "e9"],
  "turns": 1,
  "expect": "ok",
  "slots": {
    "city": "北京",
    "in_corpus": true
  },
  "omit": []
}
```

The seven Step 1 fields are immutable. Output metadata adds `idx`; resolved
values reside in `context` or `resolved`.

Repeated Step 1 rows are valid. `idx` supplies row identity independently of row
content.

## 6. Output contract

Each accepted file is one JSON object:

```json
{
  "schema_version": "cn_travel.materialized.v2",
  "metadata": {
    "idx": 42,
    "workflow": 3,
    "route": "W3-G",
    "edges": ["e1", "e2", "e4", "e8", "e10"],
    "turns": 2,
    "expect": "ok",
    "slots": {
      "path": "recommend",
      "city": "北京"
    },
    "omit": ["hotel"]
  },
  "context": {
    "user_name": "用户",
    "today": "2026-08-21",
    "current_city": {
      "name": "北京",
      "adcode": "110000",
      "weather_id": "101010100"
    },
    "start_coordinates": "116.481028,39.989643",
    "departure_window": {
      "start": "2026-08-22",
      "end": "2026-08-26"
    }
  },
  "resolved": {
    "canonical_slots": {
      "city": "北京",
      "requirements": "交通方便",
      "hotel_name": "北京国贸大酒店"
    },
    "turn_plan": [
      {
        "turn": 1,
        "intent": "hotel_reviews",
        "reveal": [],
        "conceal": ["hotel"],
        "constraints": ["deictic_hotel_reference", "review_intent_only"]
      },
      {
        "turn": 2,
        "intent": "recommend_hotels",
        "reveal": ["city", "requirements"],
        "conceal": [],
        "constraints": ["abandon_unresolved_hotel_reference"]
      }
    ],
    "assistant_asks": [
      {
        "after_turn": 1,
        "slot": "hotel",
        "text": "您想了解哪家酒店的点评呢？"
      }
    ],
    "response_policy": {
      "price_known": false,
      "weather_use": "safety_adjustments_only",
      "allowed_capabilities": ["information_only"],
      "canonical_refusal": null
    }
  },
  "visible_tool_trace": [
    {
      "ordinal": 0,
      "execution_id": "sha256:recommend-example",
      "name": "recommend_hotels",
      "arguments": {
        "location": "北京",
        "requirements": "交通方便"
      },
      "result": {
        "status": "ok",
        "source": "amap-poi/110000",
        "location": "北京",
        "hotels": [
          {
            "name": "北京国贸大酒店",
            "address": "示例地址",
            "district": "朝阳区",
            "rating": 4.8,
            "tier": "高档型",
            "tel": "",
            "lat": 39.9,
            "lng": 116.4,
            "price_cny": null
          }
        ]
      }
    },
    {
      "ordinal": 1,
      "execution_id": "sha256:review-example",
      "name": "get_hotel_reviews",
      "arguments": {
        "hotel_name": "北京国贸大酒店",
        "location": "北京"
      },
      "result": {
        "status": "ok",
        "source": "grounded-review-source",
        "hotel_name": "北京国贸大酒店",
        "rating": 4.8,
        "price_hint": null,
        "reviews": [{"text": "位置方便，服务良好。"}],
        "summary": "住客主要认可位置和服务。"
      }
    }
  ],
  "selection_trace": [],
  "provenance": {
    "run_id": "m2-20260821-1200",
    "seed": 20250914,
    "fingerprint": "72e464a20ecf6e60eaddd2da71864d87bafc723ceef6b9ba4e42ece52262c5c6",
    "source_storyboard_idx": 42,
    "candidate_id": "m2-20260821-1200-0042-a1",
    "attempt": 1,
    "created_at": "2026-08-21T12:00:00-05:00"
  }
}
```

The example values are illustrative. Materialized results must pass the project
contracts.

### 6.1 Metadata rule

`metadata` is a deep copy of the indexed Step 1 row plus `idx`:

- a selected hotel resides under `resolved.canonical_slots`;
- generated dates reside under `context` or `resolved`;
- `expect` equals the Step 1 value;
- each Step 1 city and entity retains its exact label value.

All materialized additions reside under `context`, `resolved`, tool traces, or
`provenance`.

### 6.2 Visible versus selection traces

`visible_tool_trace` is exactly what Step 3 will render as assistant tool calls and tool messages.

`selection_trace` records grounded helper calls used only to choose valid material. For example, a direct-review route may use hotel recommendations to discover a real hotel name, but the visible route contains only `get_hotel_reviews`.

Step 3 renders conversation tool calls exclusively from `visible_tool_trace`.

## 7. Run context and canonical value selection

Resolve all values before any prose is generated.

### 7.1 Frozen run context

At run start, pin:

- logical and system dates;
- user name and departure-window defaults;
- the eligible profile pool and its registry/weather-map identities;
- deterministic seed;
- selected storyboard indices.

For each candidate, derive the current city, adcode, weather ID and start
coordinates from `seed`, `idx` and `attempt`. Store the resolved context and run
fingerprint in every record, and store the same fingerprint in the manifest.
Step 3 reads its date and user context from the accepted record.

The profile `departure_window` supplies contextual defaults. The canonical W1
trip date and the Step 1 route independently govern date clarification.

The schema provides deterministic profile variation while preserving Step 1.

### 7.2 W1 canonical dates

Deterministic Step 2 logic selects:

- `start_date` in `YYYY-MM-DD`;
- `num_days` in the configured range, initially 1–5;
- exact weekday calculated from `start_date`;
- an allowed surface-date strategy for Step 3, such as exact date or a verified relative expression.

Derive choices from `seed`, `idx`, candidate `attempt`, and frozen `today`.
Accept normalized dates from this deterministic derivation and reject malformed
values explicitly.

When `date` appears in `omit`, turn 1 conceals it and the corresponding follow-up turn reveals the same canonical value. Otherwise turn 1 reveals it.

### 7.3 Exact entity fidelity

- W1 uses the exact Step 1 city. `黄石` identifies 湖北黄石; Yellowstone retains
  its distinct identity.
- W2 user-turn constraints use the exact Step 1 destination. A generic `医院`
  remains a generic nearby-hospital request.
- W3 uses the exact selected hotel name from Step 2 evidence.
- Step 3 varies grammar around an entity while retaining its exact name or an
  alias explicitly supplied by Step 2.

### 7.4 Controlled W3 requirements

For recommendation routes, Step 2 selects zero or more requirements from a controlled catalog, for example:

- tier/type;
- district or landmark proximity;
- transport convenience;
- a user budget as a request constraint.

Set `response_policy.price_known: true` exactly when captured results contain
non-null price evidence. A budget request by itself yields `price_known: false`.

## 8. Route-derived turn plans

Step 2 derives semantic user-turn plans from `route`, `omit` and Step 1 slots. Step 3 may paraphrase those plans but cannot change their intent.

Required W3 behavior:

- W3-F/G/J turn 1 is a deictic hotel-review question that conceals hotel identity.
- The fixed assistant clarification asks which hotel.
- W3-F/G/J turn 2 explicitly abandons the unresolved reference and asks for hotel recommendations in the Step 1 city.
- The supported W3 intent set is recommendation and review. Each additional
  capability requires an explicit route and tool contract.
- W3-H/I turn 2 supplies the selected exact hotel name and remains a review request.

The turn plan encodes reveal/conceal constraints for every omitted slot, giving
Step 3 validation a structured route authority.

## 9. Visible tool plans by workflow

The canonical route manifest is authoritative. At minimum it must express these rules:

### 9.1 W1

| Routes | Visible tools and statuses |
|---|---|
| W1-A/C/E/G | `search_travel_guide: ok` → `get_weather_info: ok` |
| W1-B/D/F/H | `search_travel_guide: empty`; weather suppressed |

Guide and weather locations must equal the Step 1 city. Weather arguments use the canonical Step 2 date and duration.

### 9.2 W2

| Routes | Visible tools and statuses |
|---|---|
| W2-A/C | `query_route: error` |
| W2-B/D | `query_route: ok` |

Argument rules:

- explicit Step 1 origin: use it exactly;
- omitted origin: use frozen context coordinates;
- destination: use the Step 1 destination exactly;
- `city` is always present;
- local/facility requests use the frozen profile city;
- intercity rows use the destination city as the endpoint-resolution scope and must resolve both named cities as administrative entities.

An intercity materialization exposes exactly the modes returned by `query_route`.
Step 3 describes those captured modes. A fixed row receives a deferred
disposition when the route tool cannot return a semantically valid result.

### 9.3 W3

| Route | Visible tools and statuses |
|---|---|
| W3-A/F | `recommend_hotels: ok` → `get_hotel_reviews: empty` |
| W3-B/G | `recommend_hotels: ok` → `get_hotel_reviews: ok` |
| W3-C/H | `get_hotel_reviews: empty` |
| W3-D/I | `get_hotel_reviews: ok` |
| W3-E/J | `recommend_hotels: error`; reviews suppressed |

For `get_hotel_reviews: ok`, `reviews` contains at least one review under the
declared contract.

For recommendation paths, the reviewed hotel must appear in the exact captured recommendation result. For direct-review paths, any discovery calls used to find a real hotel go into `selection_trace`, while only the review call goes into `visible_tool_trace`.

### 9.4 W4 and W5

W4 and W5 use empty visible and selection tool plans. The turn plan uses the
exact Step 1 topic and records whether Step 3 answers travel chitchat or refuses
an unrelated request.

## 10. Tool execution and caching

### 10.1 Canonical execution keys

Build a canonical execution key from:

- tool name;
- sorted-key JSON arguments;
- the tool fingerprint derived from the recorded `get_*.py` and provider-module
  hashes.

Use a concurrency-safe, per-key single-flight cache. For cacheable tools, the
first physical request or `Executor.put()` value produces the immutable
execution record, and every matching request reuses it. Local guide retrieval
reads the current corpus on each call so guide normalization changes apply
immediately; its deterministic execution ID still derives from the same key.

W3 may probe multiple distinct hotels to find the required review outcome. Each
distinct cacheable `(tool, arguments)` key yields one immutable execution record,
whether populated by a tool request or `Executor.put()`. The chosen result becomes
the visible-trace evidence.

### 10.2 Validate before accepting

Before accepting any numeric record:

1. Validate required arguments declared by `tool_schemas.json`.
2. Validate the result with `cn_travel.business_logic.contracts.validate`.
3. Validate the returned status against the route plan.
4. Validate argument/result identity fields.
5. Validate entity relevance.

Transient execution failures use the bounded retry schedule `0, 3, 10, 30, 60`
seconds. Exhausted or non-retryable infrastructure failures terminate the run.
Materialized `empty` and business `error` branches remain schema-valid tool
results.

### 10.3 Entity checks

At minimum:

- result query/location fields equal their arguments;
- administrative-city endpoints resolve to the requested canonical city;
- returned hotel name equals the review argument;
- a reviewed recommendation belongs to the captured recommendation list;
- aliases require an explicit canonical identifier or allowed-alias record.

Accept `status: ok` for an entity lookup exactly when the returned entity matches
the requested canonical identity or an explicit allowed alias.

## 11. Response policy passed to Step 3

Step 2 records machine-readable policy derived from the business prompt and captured evidence.

Initial defaults:

- `weather_use: safety_adjustments_only`: Step 3 adds safety advice and optional
  adjustments grounded in captured weather evidence; certainty matches the
  evidence.
- `price_known: false` when all `price_cny` and `price_hint` fields are null. The
  response leaves budget fit unasserted.
- `allowed_capabilities: [information_only]`: Step 3 limits offers and actions to
  information supported by the captured tool trace.
- empty/error replies use user-facing product language; knowledge-store names,
  backend names, and implementation details remain private.

## 12. Acceptance validator

Implement a pure offline validator for a materialized record. The same record
and pinned local inputs produce the same result.

Required checks:

1. UTF-8 JSON object and exact required top-level fields.
2. Filename, `metadata.idx` and storyboard array position agree.
3. The seven Step 1 metadata fields are value-identical to the input row.
4. Context and provenance fields are complete and match the run fingerprint.
5. Canonical slots have required types and ranges.
6. Turn plan length, intent, reveal/conceal order and assistant asks match the route.
7. Visible tool sequence and full status sequence match the route manifest.
8. Tool arguments pass their schemas.
9. Every result passes the declared result contract.
10. Dependencies suppress later calls after non-`ok` prerequisites.
11. Argument/result/metadata entity coupling passes.
12. W3 hotel selection and price-knowledge flags agree with captured evidence.
13. The visible trace contains exactly the calls declared by the route manifest.
14. Execution IDs are unique within a record.

Any reject-level failure prevents the numeric JSON from entering the accepted set.

### 12.1 Dataset seal invariants

Before sealing the selected dataset, validate all accepted records together:

- one execution ID maps to exactly one tool name, normalized argument object,
  and result value;
- one cacheable normalized execution key maps to exactly one execution ID and
  immutable result; and
- every selected record passes the per-record checks above.

Release acceptance requires this dataset-wide validator in addition to manifest
generation.

## 13. Failure and retry policy

Use stable error codes and dispositions.

Each `attempts.jsonl` line contains `run`, the 12-character `fingerprint`
prefix, `idx`, `route`, `attempt`, `code` and an `error` string capped at 200
characters.

### Retry transient infrastructure failures

Retry bounded timeouts, connection failures and rate limits with the executor's
fixed backoff schedule. Store the executor's `attempts` count on each cached
execution record.

### Try alternate grounded material

For W3, try recommendations in deterministic order until one produces the
route's required review outcome. `expect` retains its Step 1 value.

### Defer a row

Defer when valid material cannot satisfy the immutable Step 1 row, including:

- the available W3 hotels collectively miss the required review outcome;
- an intercity row cannot be resolved faithfully by the available route tool;
- a tool returns a stable status different from the route label.

Append a structured `DEFERRED` event for each rejected candidate attempt. After
the configured five attempts, leave the index outside the accepted-record set
and include it in `selected_deferred`.

### Reject invalid candidates

Record `GATE_REJECT` and try the next deterministic candidate for:

- missing required arguments;
- wrong planned tool/status sequence;
- metadata mutation;
- repeated contract-invalid result shapes;
- output that fails the offline validator.

After five rejected candidates, leave the row outside the accepted set and seal
the run as partial. Exceptions outside the structured deferred/gate-reject path,
including cache corruption, terminate the command before a new manifest is
sealed. Prose generation belongs to Step 3.

## 14. Resumability, concurrency and atomic writes

- Every generation invocation resumes automatically by validating existing
  numeric records against the active fingerprint and current validator.
- A valid matching record satisfies its index. An invalid or fingerprint-mismatched
  record is removed and its index returns to the pending set.
- `--adopt-fingerprint` reads the fingerprint from the existing manifest, or
  from the first numeric record when a manifest is unavailable. When the
  manifest provides materializer SHA-256 values, the next manifest carries them
  forward and adds the current source identity.
- Use single-flight locking per execution key.
- Write cache entries, accepted numeric records and the manifest through a
  sibling `.tmp` file followed by `os.replace()`.
- Append attempt events to `attempts.jsonl` under a process-local lock and write
  candidate archives under `candidates/`.
- A crash or partial temporary file leaves the index pending.
- Accepted records are immutable within one run fingerprint.
- Validated content plus a matching fingerprint establishes per-record
  completion.

## 15. Manifest contract

`manifest.json` contains:

- `schema_version`, `fingerprint`, `sealed_at` and run `state` (`partial` or
  `complete`);
- `required`, the number of rows selected by the CLI filters, and `selected`,
  the number of accepted numeric records present in the output directory;
- `sources`, containing the recorded hashes, deterministic configuration,
  profile-pool size, seed, dates and forecast horizon defined in §4;
- `counts_by_route`, `counts_by_workflow`, `counts_by_turns` and
  `counts_by_outcome`;
- `records_sha256` and `aggregate_selected_digest` for the accepted numeric
  records;
- `idx_to_candidate_id`, `identity_one_to_one` and `unique_candidate_ids`;
- `selected_deferred` and `selected_failed` counts;
- `profile_weather_mapping`, mapping each selected profile city to its adcode
  and weather ID.

`aggregate_selected_digest` is SHA-256 over the accepted records in sorted
filename order, adding `filename + NUL + sha256(file bytes) + LF` for each file.
The generator writes state `complete` exactly when the numeric-record count
equals `required`, index identity is one-to-one, candidate IDs are unique, and
no selected row is stuck after its candidate attempts. Every other completed
generation pass seals the state as `partial`. A release-qualified complete seal
also passes the dataset-wide execution-identity validator in §12.1.
`selected_deferred` counts rows that exhausted their candidate attempts;
`selected_failed` is zero because fatal generator failures terminate the command
before a new manifest is sealed.

Step 3 accepts a sealed manifest whose state is `complete` and whose
accepted-record hashes match. An explicit smoke-test mode may consume a partial
fixture.

## 16. CLI contract

Minimum target interface:

```bash
PYTHONPATH=src:. uv run --project src python data/scripts/2_materialize.py \
  --input data/1_storyboard.json \
  --output data/2_materialized \
  --workers 4 \
  --seed 20250914
```

Required options:

- `--only W1-A,W3-G`: select route tags;
- `--limit N`: select at most N rows per route;
- `--indices 1,2,3`: select exact indices;
- `--workers N`: bounded concurrency;
- `--seed N`: deterministic materialization choices;
- `--today YYYY-MM-DD`: pin date-dependent materialization;
- `--validate-only`: run read-only validation over stored artifacts;
- `--adopt-fingerprint`: resume an existing artifact set under its recorded
  fingerprint and carry forward the materializer hashes stored by its manifest;
- `--output PATH`: allow isolated smoke-test output.

Generation mode performs matching-fingerprint resume automatically.

The command exits nonzero when it encounters a fatal generator defect or when `--validate-only` finds any reject-level record.

## 17. Tests required for implementation

Create focused tests such as:

```text
data/tests/test_step2_materialize.py
data/tests/test_pipeline.py
```

Unit tests use local fixtures and mocked tool interfaces, making the suite
self-contained.

### 17.1 Golden routes

Cover all 24 route labels. Assert:

- exact turn plan;
- exact assistant clarification plan;
- exact visible tool sequence;
- full expected status sequence;
- dependency suppression;
- resolved fields required by that route.

### 17.2 Regression cases

Tests must reject:

- `query_route` without `city`;
- `get_hotel_reviews status=ok` with `reviews=[]`;
- selected/reviewed hotel mismatch;
- reviewed hotel absent from recommendations;
- changed Step 1 metadata;
- wrong city/entity or city name resolved as an unrelated POI;
- date/weekday mismatch;
- missing/extra turn or wrong reveal/conceal ordering;
- wrong tool order or dependent call after failed prerequisite;
- result-contract failure;
- selection call leaking into visible trace;
- a stored record whose provenance fingerprint differs from the active run;
- corrupt pre-existing numeric output.

### 17.3 Execution identity and resume tests

- One cacheable execution key causes at most one physical call under concurrency.
- Concurrent `Executor.run()` and `Executor.put()` calls resolve one key to one
  immutable execution record.
- Dataset-wide validation rejects an execution ID that maps to different tools,
  normalized arguments, or result values.
- W3 selected review evidence resolves to one immutable execution record.
- A valid existing record is fully validated and skipped.
- A rejected stored record returns to the pending set.
- A changed fingerprint rematerializes records under the active fingerprint;
  `--adopt-fingerprint` preserves the recorded artifact fingerprint.
- Interrupted temporary writes are ignored safely.

### 17.4 Manifest tests

- Any accepted-record mutation breaks verification.
- Manifest counts, identity flags, candidate map, record hashes and aggregate
  digest equal the accepted records.
- Step 3 rejects partial/stale/mismatched manifests.

Minimum offline command after implementation:

```bash
PYTHONPATH=src:. uv run --project src --extra test pytest -q \
  data/tests/test_step2_materialize.py \
  data/tests/test_pipeline.py
```

Live smoke tests must be separate and opt-in because they can use paid or credentialed services.

## 18. Definition of done

Step 2 is complete only when:

1. Step 1 file hashes equal their pinned inputs.
2. The materializer implements the CLI and output contracts above.
3. All 24 route fixtures pass offline.
4. Every accepted record passes the offline validator.
5. Every accepted tool result passes the project contract.
6. Each cacheable execution key resolves to one immutable execution record and
   causes at most one physical tool call.
7. Resume validates content and fingerprints rather than trusting filenames.
8. A complete manifest has `required == selected`, zero deferred/failed counts,
   one-to-one index identity, unique selected candidate IDs, and dataset-wide
   execution identity consistency.
9. Step 3 generates conversations entirely from Step 2 artifacts.
10. Every Step 2 record is stored under `data/2_materialized/`.

## 19. Additional acceptance requirements

1. **Profile variation is mandatory in v2.** It resides in Step 2 context while
   the Step 1 label remains byte-identical. Draw one internally consistent
   profile tuple — city, adcode, weather ID and centroid — from `seed`, `idx`
   and `attempt`. Pin the eligible city pool through the registry and weather-map
   hashes plus `profile_pool_size` so candidate generation is reproducible.
2. **Weather window (inclusive).** `end_date = start_date + (num_days - 1)`;
   require `end_date <= forecast_available_through`. The horizon comes from
   pinned tool/provider configuration. The adapter's 16-day duration cap and
   the provider's available-through date jointly define valid requests.
3. **§11 closes the weather-policy decision** as `safety_adjustments_only`.
   Queued records pass when their adjustments and certainty are grounded in
   weather results. The client prompt supplies guidance; §11 supplies the
   acceptance policy.
4. **W2 destination-omitted turn plan is explicit:** turn 1 reveals the exact
   origin and conceals the destination; the assistant asks for the destination;
   turn 2 reveals the exact destination. The original origin remains active
   through both turns.
5. **`--today` pins logical materialization dates.** The system date governs the
   recorded forecast horizon. The manifest records `logical_today`,
   `system_today` and `forecast_available_through`; each cached execution records
   `observed_at`, and `records_sha256` seals every accepted record. A selection
   containing W1 `ok` rows uses the system date as its logical date.
