# Training model assets

Training and data-build source models reside inside this directory. Online
inference resolves the manifest-selected copies promoted under
`src/cn_travel/`.

| Asset | Application-local path | Fetch revision |
| --- | --- | --- |
| Qwen3.5-0.8B | `base/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| LFM2.5-350M | `base/LFM2.5-350M` | `9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |
| embeddinggemma-300m | `embedding/embeddinggemma-300m` | `57c266a740f537b4dc058e1b0cda161fd15afa75` |

Fetch all three from the application directory:

```bash
make fetch-models MODEL_NAMES='qwen lfm embedding'
```

Select assets with
`uv run --project src --extra models python train/fetch_models.py qwen lfm`.
The environment variables `CN_TRAVEL_QWEN_REPO`, `CN_TRAVEL_LFM_REPO`, and
`CN_TRAVEL_EMBEDDING_REPO` select an alternate compatible snapshot source.
The matching `CN_TRAVEL_<NAME>_REVISION` variable selects its revision.

Promote the selected evaluated policy and its matching retrieval assets into
the deployable package with `make promote-runtime`. The command verifies every
recorded artifact hash before and after promotion.
