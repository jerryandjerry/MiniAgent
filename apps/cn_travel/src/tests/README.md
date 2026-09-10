# CN Travel test ownership

Each area owns the tests that verify its contracts:

- `data/tests/` verifies source corpora and the seven-stage data pipeline.
- `train/tests/` verifies training inputs, artifacts, rewards, and promotion.
- `eval/tests/` verifies the sealed evaluator and leaderboard publication.
- `src/tests/` verifies the deployable API, agent, tools, and runtime package.

## Commands

Run from the application directory:

```bash
make test-offline
make test
make test-required
make test-strict
```

The Makefile resolves the application directory from its own location. An
absolute Makefile path supports the same targets from any working directory:

```bash
make -f /path/to/cn_travel/Makefile test-offline
```

`test-offline` runs the deterministic portion of all four area suites. `test`
runs every available capability and records unavailable capabilities as skips.
`test-required` establishes every declared runtime and provider capability,
including query-dependent hotel recommendations, as a required precondition.
`test-strict` runs every available capability with
`STRICT_DEFECTS=1`, promoting expected defects to failures.

Select a file or add pytest options through the target parameters:

```bash
make test-offline \
  TESTS=eval/tests/test_eval_pipeline.py \
  PYTEST_ARGS=-q
```

## Capability markers

| Capability | Requirement |
| --- | --- |
| `rag` | Guide-tool RAG code and the promoted index and embedding model under `src/cn_travel/tool/guide/knowledge_base/` |
| `amap` | Network access to AMap |
| `live` | Configured live-provider credentials |
| `model` | PyTorch, Transformers, and the fetched Qwen base under `train/base/Qwen3.5-0.8B/` |
| `policy` | Configured policy inference endpoint |
| `reviews` | Explicitly enabled live hotel-review integration |
| `hotel_backend_real` | Query-dependent, non-placeholder hotel recommendations from the configured provider |
| `slow` | Extended runtime |

The application-root `pytest.ini` and `conftest.py` provide common capability
markers and fixtures to all four suites. The application-root Makefile invokes
the suites together.
