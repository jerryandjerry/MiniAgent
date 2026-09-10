# CN Travel data pipeline

Generated JSON artifacts are immutable pipeline outputs. The scripts in
`data/scripts/` define the reproducible seven-stage build.

| Stage | Input | Output | Script |
| --- | --- | --- | --- |
| 1. Storyboard | Business workflows and city registries | `1_storyboard.json` — 1,010 labeled scenario skeletons | `1_storyboard.py` |
| 2. Materialization | `1_storyboard.json` | `2_materialized/{idx:04d}.json` plus sealed manifest and execution cache | `2_materialize.py` |
| 3. Conversation | `2_materialized/` | `3_conversations/{idx:04d}.json` plus sealed manifest | `3_conversation.py` |
| 4. Split | `3_conversations/` | `4_split/train_validation/` — 909; `4_split/leaderboard_eval/` — 101 | `4_split.py` |
| 5. Merge | `4_split/` | `5_merged/train_validation.json` and `leaderboard_eval.json` | `5_merge.py` |
| 6. Multiturn | `5_merged/` | `6_multiturn/train_validation.json` — 3,632 prefixes; byte-identical evaluation set | `6_multiturn.py` |
| 7. World policy | Stages 2 and 5 | `7_world_policy/world_policy_episodes.jsonl` — 909 episodes and one synthetic-China world | `7_world_policy.py` |

Supporting inputs include `city_registry.json`, `city_weather_id.json`, the
scope registries, and `material_feasibility.json`. Stage 2 follows
`STEP2_MATERIALIZATION_SPEC.md`; the stage validators and `audit_report.py`
check the sealed Stage 2–3 artifacts.

Run from the application directory:

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)"
uv run --project src python data/scripts/1_storyboard.py
uv run --project src python data/scripts/2_materialize.py
uv run --project src python data/scripts/3_conversation.py
uv run --project src python data/scripts/4_split.py
uv run --project src python data/scripts/5_merge.py
uv run --project src python data/scripts/6_multiturn.py
uv run --project src python data/scripts/7_world_policy.py
uv run --project src python data/scripts/audit_report.py
```

`2_materialize.py --validate-only` performs a read-only integrity check of the
sealed Stage-2 artifacts.
