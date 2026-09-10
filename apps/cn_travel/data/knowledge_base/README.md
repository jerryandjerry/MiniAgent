# CN Travel source knowledge base

This data-owned directory contains the authoritative source corpus and the
query index built from it.

| Path | Contract |
| --- | --- |
| `travel_guides/` | Source corpus of 725 `{城市编码}_{城市名}_travel_guide.txt` files |
| `city_code_mapping.json` | Versioned city-code-to-name importer input |
| `milvus.db` | Rebuildable Milvus Lite index over the source corpus |
| `.embedding_model.json` | Embedding identity and dimension fingerprint for the index |

The index builder uses the embedding model installed by the Quick Start runtime
promotion. From the application directory, regenerate the source corpus with
`data/scripts/generate_travel_guides.py`, then build and fingerprint the source
index:

```bash
make build-kb
```

The embedding configuration in `src/cn_travel/config.yaml` governs both index
construction and runtime queries. A release records the selected index and
fingerprint hashes in `src/cn_travel/deployment_manifest.json`;
`make promote-runtime` verifies and installs that index, fingerprint, and
matching embedding model at `src/cn_travel/tool/guide/knowledge_base/`. The
deployable runtime reads the promoted copy; corpus generation and source-index
construction remain in `data/`.
