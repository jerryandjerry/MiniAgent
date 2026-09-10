---
name: build-miniagent
description: Build or extend a self-contained MiniAgent application from a client's business request and rules. Use for compiling workflows and tools, generating synthetic data, training a specialist policy, running sealed evaluation, and promoting a deployable application.
---

# Build MiniAgent

Turn the client's business brief into one independently installable application under
`apps/<app>/`.

## Inputs and ownership

The client owns the business goal and business rules. Derive the workflow design,
tools, schemas, data pipeline, model plan, evaluator, runtime, and packaging from those
inputs. Ask for a decision only when competing interpretations change business
behavior.

For a new application, render `template/` with Copier. For an existing application,
inspect its specification and artifacts before changing its contract.

## Workflow

1. **Compile the application contract.** Read
   [references/application-design.md](references/application-design.md). Produce a
   concise business brief, explicit workflow graphs, enumerated routes, typed tools,
   result states, completion behavior, and acceptance targets.
2. **Build the application boundary.** Keep the complete product in
   `apps/<app>/{data,train,eval,src}`. Give `src/` its own dependency lock and make its
   standalone project the runtime deployment unit.
3. **Compile synthetic experience.** Read
   [references/data-and-training.md](references/data-and-training.md). Generate route-
   complete scenarios, deterministic evidence, natural conversations, grouped splits,
   training views, and stateful episodes when the workflow spans user turns.
4. **Train candidate policies.** Start with a reproducible supervised baseline. Add
   preference or trajectory optimization when the measured errors justify it. Preserve
   the model-native chat template and tool-call format from training through serving.
5. **Evaluate sealed work.** Read
   [references/evaluation-and-release.md](references/evaluation-and-release.md). Roll out
   complete held-out workflows, retain transcripts, and score protocol completion,
   ordered actions, answer correctness, and latency.
6. **Promote the selected release.** Verify the winning model and knowledge artifacts,
   assemble the release manifest and selection, run verified promotion into the
   runtime package, and test a fresh copy of `src/`.

## Required invariants

- Every reachable business route has generated and validated coverage.
- The hash-bound world simulator executes every normalized tool request at least
  twice with exact agreement to replay evidence and frozen records.
- Tool arguments and `ok`, `empty`, and `error` results follow typed contracts.
- The hash-bound application conversation validator accepts every generated question
  and final response against its route semantics.
- Equivalent evidence groups remain on one side of the train/evaluation split.
- Stateful rollouts reset the same world and private episode state for every sample in
  a comparison group.
- Training, evaluation, and serving use the selected model's native message format.
- The sealed evaluator records dataset, evaluator, model, parser, and artifact identity.
- Training and evaluation manifests record their exact effective argument vectors.
- Runtime code imports only runtime-owned modules and packaged resources.

## Stage gates

Each stage finishes with a named artifact, a validation command, and a recorded hash.
Keep generated data and model outputs separate from the source that produces them.
Advance after the current stage validates. Index every valid evaluation immediately;
its status distinguishes candidates that satisfy acceptance targets.

Use the immutable artifact layout:

```text
data/artifacts/manifests/<dataset>.json
data/artifacts/<dataset>/
train/runs/<run>/training-manifest.json
eval/results/runs/<evaluation>/evaluation-manifest.json
eval/releases/<release>/release-manifest.json
eval/leaderboard.json
```

The dataset payload contains both the training and sealed evaluation splits. Keep
manifests and the leaderboard in version control; store bulky payloads and model
artifacts in application-owned artifact storage.

From the repository root, use
`python3 skills/build-miniagent/scripts/compile_routes.py` to enumerate routes from a
Mermaid workflow and `python3 skills/build-miniagent/scripts/validate_scaffold.py` to
check a newly rendered application.

External training, paid services, publication, and production deployment remain
explicit execution steps within the user's authorized scope.
