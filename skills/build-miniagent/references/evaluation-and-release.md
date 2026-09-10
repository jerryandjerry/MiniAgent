# Evaluation and release contract

## Sealed evaluation

Create the held-out split and evaluator identity before candidate training. The
evaluator runs complete workflows through a declared serving channel and a frozen,
deterministic world.

Score dimensions include:

- protocol-complete goal resolution;
- exact ordered workflow trace;
- ordered tool-call and normalized-argument agreement;
- final-answer correctness required by the business contract;
- invalid, extra, premature, and missing actions;
- response latency and evaluation errors.

Retain the transcript, normalized actions, per-case score, workflow slice, latency, and
error for every case. Aggregate only runs that pass the evaluator's admission checks.

Write each immutable result to
`eval/results/runs/<evaluation>/evaluation-manifest.json` and append its comparable
summary to `eval/leaderboard.json`.

Each leaderboard row records:

- candidate identifier and version;
- a comparability key derived from the application, sealed dataset,
  evaluator source and configuration, and complete serving protocol;
- aggregate scores and status; and
- the immutable evaluation-manifest reference.

The referenced evaluation manifest binds the candidate artifacts, sealed case
evidence, evaluator artifacts, protocol, and effective invocation.

## Selection and promotion

Select a release using declared accuracy, latency, runtime-size, and operational
targets. Promotion is an explicit build step:

1. verify the selected training and evaluation evidence;
2. assemble `eval/releases/<release>/release-manifest.json` and
   `eval/results/selected-release.json` with source and destination hashes;
3. run promotion to verify the transitive lineage and copy the selected policy,
   contracts, and required knowledge-base artifacts into `src/<package>/`;
4. resolve every runtime path relative to the installed package;
5. copy `src/` into a fresh temporary directory and install it from its own project;
6. run readiness, API, CLI, tool-contract, and representative end-to-end checks.

The promoted package exposes its model identity, artifact verification state, and
readiness status. A rollback selects a previously verified release manifest and repeats
the same standalone validation.
