# Data and training contract

## Synthetic-data compiler

Use explicit stages with immutable outputs:

1. **Storyboard:** allocate route-balanced structural scenarios.
2. **Materialization:** resolve entities, dates, private state, tool calls, and
   deterministic tool evidence.
3. **Conversation:** generate natural user and assistant language in the canonical
   function-calling format while replaying the frozen plan and tool evidence
   value-for-value.
4. **Grouped split:** assign equivalent evidence groups wholly to training or sealed
   evaluation while retaining workflow coverage.
5. **Merge:** produce stable ordered datasets and manifests.
6. **Training views:** render the canonical conversations into the model-native
   message format and derive the decision units required by the chosen objective.
7. **World policy:** compile complete stateful episodes for trajectory learning.

Every stage records its input identity, configuration, output count, validation result,
and content hash. A validation mode performs read-only checks.

Implement `data/conversation_validator.py` as the application-specific semantic gate
for follow-up intent and final response requirements. The data manifest binds its exact
source, and later stages replay that snapshot.

Write each immutable dataset manifest to
`data/artifacts/manifests/<dataset>.json`. Store its training and sealed evaluation
payloads together under `data/artifacts/<dataset>/`.

## Stateful simulation

Use one versioned synthetic domain world, one generic simulated-user implementation,
and one private state instance per episode. A rollout group receives identical reset
copies of the world version and person state. Model sampling is the source of variation
inside the group.

The world is deterministic, schema-valid, internally consistent, and stable for the
same normalized request. Its hash-bound simulator exposes
`execute(tool_id, normalized_arguments)`. Dataset validation invokes that entrypoint
at least twice for every normalized query and requires exact agreement with replay
evidence and frozen records. The user simulator reveals private facts only in response
to eligible agent actions.

## Training profiles

Record each candidate in `train/runs/<run>/training-manifest.json` with:

- base-model identity and the typed base artifacts used by the run;
- the immutable dataset manifest;
- the exact named training split;
- trainer source and environment artifacts;
- the effective launcher arguments and hashed method configuration;
- method, seed, optimizer-step and rollout totals, wall time, and metrics;
- typed outputs, including the deployable policy and any tokenizer, chat-template,
  parser, runtime-adapter, checkpoint, log, or metrics artifacts produced by the
  run; and
- the final deployable policy hash.

Build a supervised baseline first. Train only assistant decision tokens appropriate to
the target. Evaluate that baseline before selecting further optimization.

For trajectory optimization, score completed episodes in the synthetic environment.
Make terminal goal completion the eligibility gate for success credit, and use action
quality to distinguish complete trajectories. Record KL, entropy, reward, policy loss,
and completion diagnostics when the trainer exposes them.

Training environments are application-local profiles. Runtime, supervised training,
and trajectory training may use separate dependency locks or container images while
sharing immutable data and model identities.
