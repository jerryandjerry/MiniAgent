# Operations

Each MiniAgent application owns its environments, commands, artifacts, and
deployment. Run operational commands from `apps/<application>/`.

## Environments

### Runtime environment

`src/pyproject.toml` and `src/uv.lock` define the installable runtime. Optional
dependency groups may cover tests, retrieval, local inference, and provider
integrations. `src/.python-version` fixes the interpreter version.

```bash
make install
```

### Training environments

Each training implementation under `train/` records its framework-specific
environment or container, Python interpreter, accelerator requirements, and
lock information. Separate trainers can use independent dependency stacks while
sharing the same frozen application data and contracts.

### Configuration

`src/.env.example` lists configuration names and safe examples. Deployment
injects values through the target platform's configuration and secret stores.
The release manifest declares required environment names and hash-binds runtime
code plus checked-in configuration defaults.

## Stable commands

```bash
make help
make install
make validate
make data-build
make train
make eval
make promote-runtime
make test
make test-standalone
make serve
make chat
```

`make help` lists the stable commands and the configured Python launchers,
training and evaluation argument variables, and service bind variables.

## Data build

```bash
make validate
make data-build
```

The build records:

- source specification and contract hashes;
- generator code revision and configuration;
- random seeds and synthetic-world identity;
- artifact counts and hashes;
- validation results; and
- grouped train/holdout membership.

A successful build leaves an immutable dataset manifest and reproducible
training inputs. The dataset payload, including its training and sealed evaluation
splits, lives at `data/artifacts/<dataset>/`; its immutable manifest lives at
`data/artifacts/manifests/<dataset>.json`.

## Training

```bash
make train
```

The application selects a default run configuration. Additional configurations
remain addressable through variables documented by `make help`.

Each `train/runs/<run>/training-manifest.json` records:

- base-model identity and typed artifacts used by the run;
- the dataset manifest and exact named training split;
- trainer source and environment artifacts;
- the effective launcher arguments and hashed training configuration;
- method, seed, run summary, and metrics; and
- typed output artifacts and the final deployable policy hash.

Checkpointing and automatic resume protect accelerator runs. Training artifacts
remain under the application `train/` area until release promotion.

## Remote accelerator workflow

1. Validate and seal application data locally.
2. Transfer the application or immutable training bundle to the accelerator.
3. Install the recorded trainer environment.
4. Run the application training command with the selected configuration.
5. Preserve logs, metrics, final model artifacts, configuration, and hashes.
6. Run evaluation through the declared serving protocol.
7. Transfer the final source snapshot, selected model artifacts, evaluation
   evidence, and run manifest back to the application.
8. Shut down the accelerator after all artifacts are verified locally.

## Evaluation

```bash
make eval
```

The evaluator records a result as soon as each candidate completes at
`eval/results/runs/<evaluation>/evaluation-manifest.json`. A row in
`eval/leaderboard.json` contains:

- evaluation identifier and creation time;
- candidate identifier and version;
- comparability key;
- status and aggregate metrics; and
- immutable evaluation-manifest reference.

The evaluation manifest binds the sealed dataset, evaluator, candidate
artifacts, protocol, per-case evidence, effective invocation, summary counts,
and metrics.

Comparable entries share the same sealed dataset, evaluator, normalization,
protocol contract, and metric definitions.

## Promotion

```bash
make promote-runtime
make test
make test-standalone
```

Release selection prepares `eval/releases/<release>/release-manifest.json` and
`eval/results/selected-release.json`. Promotion consumes those immutable inputs
and performs these checks:

1. Select a completed leaderboard entry that satisfies the application
   thresholds.
2. Assemble an immutable release manifest and
   `eval/results/selected-release.json` from that exact entry and its verified
   training and evaluation artifacts.
3. Run `make promote-runtime` to verify the complete transitive artifact chain
   and copy the selected artifacts into `src/`.
4. Verify the installed deployment manifest and complete `src/` inventory.
5. Run `make test` and `make test-standalone`.

The promoted `src/` directory is the deployment artifact.

## Serving

```bash
make serve
```

For interactive verification:

```bash
make chat
```

The application README defines endpoint paths, request and response schemas,
session behavior, process topology, readiness checks, and target-specific
packaging. Small policies can use CPU, edge, local GPU, or hosted inference
while preserving the evaluated application protocol.

## Release and rollback

Each release receives an immutable version and deployment manifest. Keep the
previous verified bundle available. Rollback selects that bundle, verifies its
manifest, runs readiness checks, and restores traffic according to the target
platform's deployment procedure.

Operational telemetry covers request latency, workflow completion, tool status,
invalid model actions, retries, and release identity. Logs apply the
application's privacy and retention rules.
