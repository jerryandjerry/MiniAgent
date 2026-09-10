# Getting Started

This guide turns one business brief into a self-contained MiniAgent
application.

## 1. Create the application

Install Git, [`uv`](https://docs.astral.sh/uv/), Make, Bash, `awk`, `tar`, and
`mktemp`, then run:

```bash
git clone https://github.com/jerryandjerry/MiniAgent.git
cd MiniAgent
uvx copier copy ./template apps/order_status
cd apps/order_status
```

Copier asks for:

| Variable | Meaning | Example |
| --- | --- | --- |
| `app_name` | Human-readable application name | `Order Status` |
| `package_name` | Importable Python package | `order_status` |
| `description` | One-sentence business outcome | `Report order delivery status` |
| `author_name` | Application owner | `Example Company` |
| `python_version` | Runtime Python version | `3.12` |

The generated directory contains the four application areas: `data/`,
`train/`, `eval/`, and the deployable `src/` project.

## 2. Write the business brief

Complete `BUSINESS_BRIEF.md` with business language. Define:

- the outcome the application delivers;
- supported user requests;
- business rules and required decisions;
- information already present in account or request context;
- information the agent may ask the user to provide;
- business systems, APIs, datasets, and knowledge sources;
- success, empty-result, error, completion, and refusal behavior; and
- business constraints and priorities that affect the expected outcome.

A compact brief is sufficient:

```markdown
# Order Status

## Goal
Report the current delivery state of a customer's order.

## Requests
- Locate an order by order ID.
- Use the authenticated customer's active order when applicable.
- Explain a delay from recorded delivery events.

## Rules
- Ask for the order ID when account context cannot identify one order.
- Return the recorded order state and delivery events.
- Report an unknown order as an empty result.

## Systems
- Order API
- Delivery-events API

## Acceptance
- Supported requests return the correct recorded delivery state.
- Responses meet the product latency target.
```

## 3. Start the coding agent

Give the coding agent this instruction:

> Build the MiniAgent application in `apps/order_status` from
> `BUSINESS_BRIEF.md` by following the repository-root skill at
> `../../skills/build-miniagent/SKILL.md`. Compile the
> business specification and workflow routes,
> generate and validate the synthetic data, train the specialist model, run the
> sealed evaluation, promote the selected release, and verify the standalone
> runtime.

The coding agent first produces the compiled application specification:

- workflow state graphs;
- enumerated route tables;
- slots and follow-up questions;
- tool schemas and deterministic result contracts;
- scenario coverage and acceptance criteria.

Review this specification as the business-policy checkpoint. Approval fixes the
meaning of the product before generation and training begin.

## 4. Build and validate the data

From the generated application directory:

```bash
make install
make validate
make data-build
```

`make validate` checks the compiled specification and contracts. `make
data-build` creates route-balanced scenarios, deterministic tool evidence,
natural conversations, validated trajectories, and the grouped train/holdout
split described by the application.

The data gate requires:

- coverage for every workflow route;
- schema-valid tool calls and results;
- exact route replay for every trajectory;
- semantic validation of every follow-up and final response;
- hash-bound synthetic-world simulator replay at least twice for every
  normalized query, with exact agreement between replay evidence and frozen
  records; and
- a sealed holdout isolated from related training evidence.

The immutable dataset layout is:

```text
data/artifacts/manifests/<dataset>.json   dataset identity and provenance
data/artifacts/<dataset>/                 train and sealed evaluation payloads
```

## 5. Train and evaluate

```bash
make train
make eval
```

The application chooses its model and training framework. The run manifest at
`train/runs/<run>/training-manifest.json` binds the input dataset, named training split, base model,
trainer, effective invocation, configuration, metrics, and typed output
artifacts.

The evaluator runs complete workflows on the sealed holdout and records:

- goal completion;
- exact workflow trace;
- ordered tool-call accuracy;
- invalid or extra actions;
- response latency; and
- the transcript behind every score.

Each evaluation writes
`eval/results/runs/<evaluation>/evaluation-manifest.json`. Each leaderboard row
records the candidate, metrics, comparability key, and immutable manifest
reference; the manifest binds the evaluator, sealed dataset, candidate
artifacts, protocol, evidence, and effective invocation.

## 6. Promote the selected release

Select one passing leaderboard entry. Assemble
`eval/releases/<release>/release-manifest.json` and
`eval/results/selected-release.json` from that entry and its verified dataset,
training, evaluation, policy, contract, and runtime artifacts. Then run:

```bash
make promote-runtime
make test
make test-standalone
```

Promotion verifies and copies the selected model and required knowledge-base
artifacts into `src/`. The release manifest binds those artifacts to the model
adapter, parser, contracts, and runtime configuration.

The release manifest is the immutable release record. Promotion verifies its
complete lineage before installing its declared artifacts.

`make test-standalone` copies `src/` into a fresh temporary directory, installs
the copied project from its application-owned lockfile, and verifies the deployment
boundary.

## 7. Run the application

```bash
make serve
```

For an interactive terminal:

```bash
make chat
```

The generated application README documents its endpoint, request schema,
environment variables, policy-server arrangement, and deployment target.

## Command reference

| Command | Contract |
| --- | --- |
| `make help` | List application commands and arguments |
| `make install` | Install the application-owned environment |
| `make validate` | Validate specification and contracts |
| `make data-build` | Build and validate synthetic data |
| `make train` | Run the selected training configuration |
| `make eval` | Run the sealed evaluator |
| `make promote-runtime` | Verify and install the selected release in `src/` |
| `make test` | Run the application test suite |
| `make test-standalone` | Verify `src/` as an independent project |
| `make serve` | Start the application API |
| `make chat` | Start the interactive client |

The completed [CN Travel application](../apps/cn_travel/README_CN%20TRAVEL.md)
is the reference for a fully implemented data, training, evaluation, and
runtime lifecycle.
