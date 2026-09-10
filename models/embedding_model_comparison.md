# Embedding Model Comparison

## Core Specification and Benchmark Comparison

| Feature / Benchmark | **Qwen3-Embedding-0.6B** | **EmbeddingGemma-300M** | **BGE-M3** |
|---|---:|---:|---:|
| **Parameters** | 595M | **308M** | 568M |
| **Reported model memory** | ~2272 MB | **~578 MB** | ~2167 MB |
| **Max input sequence length** | **32K tokens** | 2,048 tokens | 8,192 tokens |
| **Default / max dense vector dim** | 1024 | **768** | 1024 |
| **Reduced vector dimensions** | **32–1024 via MRL** | 128 / 256 / 512 / 768 via MRL | No native MRL |
| **Float32 storage per dense vector** | ~4 KB | **~3 KB** | ~4 KB |
| **MTEB Multilingual v2 Overall** ↑ | **64.34 🥇** | 61.15 | 59.56 |
| **Multilingual Retrieval** ↑ | **64.65 🥇** | 62.49 🥈 | 54.60 |
| **Multilingual Reranking*** ↑ | 61.41 | **63.25 🥇** | 62.79 🥈 |
| **Multilingual STS** ↑ | **76.17 🥇** | 74.73 | 74.12 |
| **MTEB English v2 Overall** ↑ | **70.70** | 69.67 | — |
| **MTEB Code** ↑ | **74.57** | 68.14 | — |
| **Dense retrieval** | ✅ | ✅ | ✅ |
| **Sparse retrieval** | ❌ | ❌ | **✅** |
| **Multi-vector / ColBERT-style** | ❌ | ❌ | **✅** |
| **Instruction-aware / task prompts** | ✅ | ✅ | Not required for normal retrieval |
| **100+ languages** | ✅ | ✅ | ✅ |
| **Dedicated matching reranker** | **Qwen3-Reranker-0.6B** | None | **bge-reranker-v2-m3** |
| **Main advantage** | **Best dense retrieval + longest context** | **Best size / memory efficiency** | **Dense + sparse + multi-vector hybrid retrieval** |

\* `Reranking` in the benchmark table measures how well the **embedding model itself** performs on embedding-based reranking tasks. It is not the score of a dedicated reranker such as Qwen3-Reranker-0.6B.

## Sequence-Length Limitation

Every embedding model has a maximum input sequence length for **one input text/chunk**:

| Model | Maximum input |
|---|---:|
| **Qwen3-Embedding-0.6B** | **32K tokens** |
| **BGE-M3** | **8,192 tokens** |
| **EmbeddingGemma-300M** | **2,048 tokens** |

If a chunk exceeds the model's maximum sequence length, the inference library normally either **truncates the input** or rejects it, depending on configuration.

## Vector Dimension

The vector dimension controls the size of each stored dense embedding.

For float32 vectors:

- **1024 dimensions** ≈ 4 KB per vector
- **768 dimensions** ≈ 3 KB per vector
- **512 dimensions** ≈ 2 KB per vector
- **256 dimensions** ≈ 1 KB per vector
- **128 dimensions** ≈ 0.5 KB per vector

`Qwen3-Embedding-0.6B` and `EmbeddingGemma-300M` support **MRL (Matryoshka Representation Learning)**, meaning you can intentionally truncate the embedding to a smaller dimension to reduce vector-database storage and search cost while retaining much of the model's retrieval quality.

BGE-M3's normal dense embedding is 1024-dimensional. Its optional ColBERT-style mode is different: it can generate **multiple token-level vectors per text**, so storage can be much larger than ordinary one-vector-per-chunk dense retrieval.

## Practical Recommendation

| Goal | Pick |
|---|---|
| **Highest normal dense RAG retrieval quality** | **Qwen3-Embedding-0.6B** |
| **Best memory / compute efficiency** | **EmbeddingGemma-300M** |
| **Need long chunks / documents** | **Qwen3-Embedding-0.6B** |
| **Dense + keyword + fine-grained hybrid retrieval** | **BGE-M3** |
| **Simplest strong pipeline** | **Qwen3-Embedding-0.6B → Qwen3-Reranker-0.6B** |
| **Lightest strong pipeline** | **EmbeddingGemma-300M → Qwen3-Reranker-0.6B** |
| **Most feature-rich retrieval pipeline** | **BGE-M3 dense+sparse(+multi-vector) → reranker** |

## Model Links

- Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen3-Reranker-0.6B: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- EmbeddingGemma-300M: https://huggingface.co/google/embeddinggemma-300m
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- BGE-Reranker-v2-M3: https://huggingface.co/BAAI/bge-reranker-v2-m3

## Sources

- EmbeddingGemma paper (common benchmark table, parameter count, reported memory usage):
  https://arxiv.org/abs/2509.20354
- Qwen3-Embedding-0.6B:
  https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- BGE-M3:
  https://huggingface.co/BAAI/bge-m3
