# Reranker Comparison

| Model | Params | Max input | Architecture | MTEB-R ↑ | MMTEB-R ↑ | MLDR ↑ | MTEB-Code ↑ | License |
|---|---:|---:|---|---:|---:|---:|---:|---|
| **Qwen3-Reranker-0.6B** | 0.6B | **32K** | Decoder LM reranker | **65.80** | **66.36** | **67.28** | **73.42** | Apache-2.0 |
| **GTE-Multilingual-Reranker-Base** | **306M** | 8K | Encoder cross-encoder | 59.51 | 59.44 | 66.33 | 54.18 | Apache-2.0 |
| **BGE-Reranker-v2-M3** | ~568M | 8K | XLM-R encoder cross-encoder | 57.03 | 58.36 | 59.51 | 41.38 | Apache-2.0 |

## Recommendation

| Priority | Choice |
|---|---|
| **Best quality** | **Qwen3-Reranker-0.6B** |
| **Best speed / smallest model** | **GTE-Multilingual-Reranker-Base** |
| **Stay in BGE ecosystem** | **BGE-Reranker-v2-M3** |

## Suggested Pipelines

```text
Qwen3-Embedding-0.6B
        ↓
Qwen3-Reranker-0.6B
```

```text
EmbeddingGemma-300M
        ↓
Qwen3-Reranker-0.6B
```

## Model Links

- Qwen3-Reranker-0.6B: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- GTE-Multilingual-Reranker-Base: https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base
- BGE-Reranker-v2-M3: https://huggingface.co/BAAI/bge-reranker-v2-m3
