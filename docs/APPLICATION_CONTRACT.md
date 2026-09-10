# Application Contract

Every MiniAgent application is a self-contained product under
`apps/<application>/`.

## Required structure

```text
apps/<application>/
├── .gitignore
├── BUSINESS_BRIEF.md
├── README.md
├── Makefile
├── pytest.ini
├── data/
│   ├── README.md
│   ├── config.json
│   ├── conversation_validator.py
│   ├── pipeline.py
│   ├── specification/
│   │   ├── application.json
│   │   ├── schemas/
│   │   ├── tools/<tool>.json
│   │   └── workflows/<workflow>.json
│   ├── scripts/
│   │   ├── _artifacts.py
│   │   ├── _contract.py
│   │   ├── _dataset.py
│   │   ├── _evaluation.py
│   │   ├── _training.py
│   │   ├── build.py
│   │   └── validate.py
│   ├── artifacts/
│   │   ├── manifests/
│   │   └── <dataset>/
│   ├── tests/
│   └── reproducibility/
├── train/
│   ├── config.json
│   ├── configurations/
│   ├── run.py
│   ├── runs/<run>/training-manifest.json
│   ├── tests/
│   └── README.md
├── eval/
│   ├── config.json
│   ├── configurations/
│   ├── evaluate.py
│   ├── promote.py
│   ├── results/
│   │   ├── selected-release.json
│   │   └── runs/<evaluation>/evaluation-manifest.json
│   ├── releases/<release>/release-manifest.json
│   ├── leaderboard.json
│   ├── tests/
│   └── reproducibility/
└── src/
    ├── README.md
    ├── pyproject.toml
    ├── uv.lock
    ├── .python-version
    ├── .env.example
    ├── tests/
    └── <package>/
        ├── __init__.py
        ├── __main__.py
        ├── api.py
        ├── cli.py
        ├── agent.py
        ├── config.py
        ├── release_verifier.py
        ├── deployment_manifest.json
        ├── business_logic/
        ├── service/
        ├── tool/
        └── model/
```

`make install` creates `src/uv.lock`. Release assembly creates
`eval/results/selected-release.json` and the selected immutable run and release
records.

Applications may add files and subdirectories while preserving these ownership
boundaries and the stable command surface.

## Business brief

`BUSINESS_BRIEF.md` is the client-authored source. It defines:

| Field | Contract |
| --- | --- |
| Goal | Business outcome delivered to the user |
| Requests | Supported intents and representative user language |
| Rules | Required decisions, order, branches, and completion behavior |
| Context | Values available from the request, account, session, or environment |
| User input | Values the agent may collect through follow-up questions |
| Systems | APIs, databases, documents, and knowledge sources available to the application |
| Constraints | Privacy, safety, compliance, cost, and latency requirements |

The coding agent derives the technical architecture, workflow and tool
contracts, synthetic data, model plan, measurable acceptance criteria,
evaluator, runtime, and deployment design from this business input.

## Compiled application specification

`README.md` is the human-readable implementation specification derived by the
coding agent. `data/specification/application.json` is its machine-readable
application contract. Together they contain:

- workflow names and supported requests;
- Mermaid state graphs with typed decision, question, tool, and terminal nodes;
- deterministic route tables;
- slot definitions and follow-up wording constraints;
- tool schemas, result envelopes, and dependency rules;
- synthetic scenario and world policy;
- data pipeline inputs, outputs, formats, examples, and validation gates;
- training configurations and run records;
- evaluation protocol, metrics, leaderboard, and promotion thresholds; and
- runtime setup, API, CLI, configuration, and deployment contract.

The machine-readable artifacts conform to the repository schemas:

| Artifact | Root schema |
| --- | --- |
| Application specification | `contracts/application.schema.json` |
| Workflow | `contracts/workflow.schema.json` |
| Tool | `contracts/tool.schema.json` |
| Dataset manifest | `contracts/dataset-manifest.schema.json` |
| Training run | `contracts/training-manifest.schema.json` |
| Evaluation result | `contracts/evaluation-manifest.schema.json` |
| Promoted release | `contracts/release-manifest.schema.json` |

## Workflow contract

Each workflow defines its `application_id`, `workflow_id`, name, goal, slots,
initial state, typed states, guarded transitions, and enumerated terminal
routes. The exact representation is defined by
`contracts/workflow.schema.json`.

Route enumeration is deterministic. Every generated sample carries its workflow
and route identifiers, and validation replays the observed trajectory against
that declared route.

## Tool contract

Every model-visible tool defines:

- a stable name and purpose;
- typed, schema-constrained arguments;
- ordered executable argument-normalization operations;
- one result schema with explicit status;
- deterministic synthetic-world behavior;
- production provider behavior;
- prerequisite and same-round dependency rules; and
- timeout, retry, idempotency, and error semantics where applicable.

The result envelope distinguishes normal data, normal absence, and execution
failure:

```json
{"status": "ok", "data": {}}
{"status": "empty", "data": null}
{"status": "error", "error": {"code": "...", "message": "..."}}
```

## Data contract

Every dataset record has this stable shape:

- `episode_id`, `workflow_id`, and `route_id`;
- `initial_context` and `private_state`;
- `trajectory`, containing initial slots, ordered route transitions, typed
  actions, terminal state, and goal completion;
- `tool_evidence`, containing ordered arguments, normalized arguments, and
  frozen results;
- `conversation` in the canonical function-calling format, replayed exactly
  against the trajectory and frozen tool evidence; and
- `contract_version` and `generator_version`.

The dataset payload lives at `data/artifacts/<dataset>/` and contains its
training and sealed evaluation splits. Its immutable manifest lives at
`data/artifacts/manifests/<dataset>.json` and records counts, hashes, inputs,
generator revisions, the hash-bound semantic conversation validator, exact
world-query replay evidence, validation status, and group boundaries. Training derives
views from these records. Both training and evaluation partitions cover every
declared workflow route. Training views render the canonical conversation into
the selected model's native message and action format.

The canonical tool exchange uses an assistant function call followed by the
matching tool result:

```json
{"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"catalog.lookup","arguments":"{\"query\":\"alpha\"}"}}]}
{"role":"tool","tool_call_id":"call-1","content":"{\"status\":\"ok\",\"data\":{}}"}
```

Each later stage writes one immutable record:

```text
train/runs/<run>/training-manifest.json
eval/results/runs/<evaluation>/evaluation-manifest.json
eval/releases/<release>/release-manifest.json
eval/leaderboard.json
```

## Runtime contract

The deployable `src/` project owns:

- API and CLI entry points;
- session and conversation state;
- agent orchestration and bounded tool dispatch;
- business-policy and tool contracts;
- service facades and provider adapters;
- native model adapter and action parser;
- selected policy and knowledge-base artifacts;
- configuration and readiness checks; and
- runtime and standalone tests.

The dependency direction is:

```text
API / CLI -> Agent -> Service -> Tool / Model -> Provider
```

Runtime modules import within `src/`. Offline `data/`, `train/`, and `eval/`
consume the authoritative contracts under `data/specification/` and recorded
artifacts. Promotion copies their verified snapshots into `src/`.

## Release contract

The deployment manifest identifies:

- application and release versions;
- the selected policy weights, action parser, and runtime adapter;
- application, workflow, and tool-contract snapshots;
- source training run, dataset, evaluator, sealed split, and exact leaderboard
  entry;
- complete runtime source, dependency, contract, test, and configuration
  inventory;
- knowledge-base, embedding, tokenizer, and chat-template artifacts used by the
  selected application; and
- SHA-256 and byte count for every promoted artifact.

Promotion verifies the manifest before copying artifacts into `src/`. Readiness
verifies the same identities and declared Python constraint before serving traffic.

## Stable command surface

Every application implements:

```text
help  install  validate  data-build  train  eval
promote-runtime  test  test-standalone  serve  chat
```

The application Makefile owns command orchestration and path resolution.
