# MiniAgent Recipe

The recipe compiles client-owned business logic into a trained, evaluated, and
deployable specialist application. Every stage produces explicit artifacts and
passes a gate before the next stage begins.

## Inputs and ownership

| Owner | Inputs and responsibilities |
| --- | --- |
| Client | Business request and business rules, including relevant context and systems |
| Coding agent | Workflows, tools, acceptance criteria, data, training, evaluation, runtime, deployment, and reproducibility |

The root repository supplies the method, template, schemas, and coding-agent
workflow. Each generated application owns its implementation and artifacts.

## Stage 1: Compile the business policy

**Input:** `BUSINESS_BRIEF.md`

**Outputs:**

- supported and out-of-scope request classes;
- workflow state graphs;
- slots and ordered follow-up questions;
- result-dependent branches and termination rules;
- deterministically enumerated route tables; and
- route-level acceptance criteria.

**Gate:** every workflow terminates, every graph edge belongs to a route, every
route has a unique identifier, and the client approves the compiled behavior.

## Stage 2: Define application contracts

**Input:** approved workflows and connected-system requirements

**Outputs:**

- typed tool parameters and model-visible schemas;
- `ok`, `empty`, and `error` result envelopes;
- tool dependency and dispatch rules;
- conversation state and model action grammar;
- service and provider interfaces;
- evaluation contract; and
- runtime API and CLI contract.

**Gate:** contracts represent every workflow edge, tool execution is bounded by
an allowlist, and invalid or premature actions produce deterministic outcomes.

## Stage 3: Generate synthetic experience

**Input:** routes, contracts, scenario distributions, and synthetic world

**Outputs:**

1. route-balanced scenario specifications;
2. schema-valid, deterministic tool evidence;
3. natural conversations that replay the frozen evidence;
4. application-validated questions, responses, and complete trajectories;
5. grouped training and sealed holdout partitions; and
6. manifests binding every artifact to its inputs and generator revision.

The dataset payload, including its sealed evaluation split, lives at
`data/artifacts/<dataset>/`. Its immutable manifest lives at
`data/artifacts/manifests/<dataset>.json`.

Each route fixes the workflow structure. Generation varies user wording,
entities, values, and final prose within that route.

For stateful workflows, the application defines:

- one internally consistent synthetic domain world;
- one generic synthetic-user policy;
- one private person or scenario state per episode; and
- identical state resets for every sampled rollout of an episode.

**Gate:** all routes meet their coverage targets, every trajectory and conversation
passes structural and semantic replay, the hash-bound world simulator reproduces
every normalized query at least twice with exact results, and related
evidence remains within one data partition.

## Stage 4: Train the specialist policy

**Input:** one named validated training split, native model format, and sealed run
configuration

**Outputs:**

- selected sub-1B base model and immutable revision;
- training-ready views derived from complete conversations;
- checkpoints and final adapter or model;
- optimization metrics; and
- `train/runs/<run>/training-manifest.json` with code, data, model,
  configuration, and artifact hashes.

The application selects the learning method that matches its behavior. Format
supervision teaches the native conversation and action protocol. Trajectory
optimization can refine multi-turn goal completion, ordered actions, recovery,
and efficiency. Training and serving share the same chat template and action
parser.

**Gate:** the final artifact is loadable, its identity is recorded, its parser
matches the runtime, and the run can resume or reproduce from recorded inputs.

## Stage 5: Evaluate complete workflows

**Input:** sealed holdout, frozen evaluator, candidate release, and declared
serving protocol

**Outputs:**

- one transcript and score record per case;
- goal-completion and workflow-trace metrics;
- ordered tool-call and argument metrics;
- invalid-action and extra-action counts;
- latency distribution;
- evaluator, dataset, model, and artifact identities; and
- `eval/results/runs/<evaluation>/evaluation-manifest.json`; and
- a comparable entry in `eval/leaderboard.json`.

The evaluator treats follow-up questions, tool calls, result-dependent rounds,
and valid final responses as parts of one completed task.

**Gate:** every case completes through the declared protocol, all evidence is
preserved, and the candidate satisfies the application's promotion thresholds.

## Stage 6: Promote and deploy

**Input:** selected leaderboard candidate and its verified artifacts

**Outputs:**

- runtime model and knowledge-base artifacts under `src/`;
- `eval/releases/<release>/release-manifest.json` binding hashes, contracts,
  parser, and configuration;
- installable application package;
- API and CLI launch points; and
- standalone verification record.

**Gate:** manifest verification passes, `make test` passes, `make
test-standalone` passes from a fresh copy of `src/`, and runtime behavior matches
the evaluated protocol.

## Completion criteria

An application is complete when:

- the business policy is explicit and route-complete;
- synthetic data is deterministic, validated, and reproducible;
- the trained policy is identified by immutable artifacts;
- sealed evaluation evidence supports the reported accuracy and latency;
- the promoted runtime contains every production dependency and artifact; and
- a fresh environment can install and run `src/` independently.
