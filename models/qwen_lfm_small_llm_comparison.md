# Small LLM Comparison: Qwen3.5-0.8B vs LFM2.5-350M

## Core Specifications

| Feature | **Qwen3.5-0.8B** | **LFM2.5-350M** |
|---|---:|---:|
| **Parameters** | 0.8B | **350M** |
| **Type** | Multimodal causal LM | Text-only causal LM |
| **Architecture** | **Hybrid linear attention + standard attention** | **Hybrid convolution + standard attention** |
| **Layer layout** | 24 layers: 18 Gated DeltaNet + 6 Gated Attention | 16 layers: 10 double-gated convolution + 6 GQA |
| **Mamba / SSM** | ❌ | ❌ |
| **Linear attention** | **✅ Gated DeltaNet** | ❌ |
| **Standard attention** | ✅ | ✅ GQA |
| **Native context length** | **262,144 tokens** | 32,768 tokens |
| **Vision input** | **✅** | ❌ |
| **Tool / function calling** | ✅ | ✅ |
| **Languages** | **201 languages and dialects** | 9 listed languages |
| **Training / intended strength** | General reasoning, multimodal, agents, multilingual | Structured output, extraction, tool use, edge deployment |
| **Main weakness at this size** | More compute/memory than LFM | Weak knowledge-intensive tasks and programming |
| **Main advantage** | **Much stronger knowledge/reasoning + huge context + vision** | **Tiny, fast, excellent instruction following / tool use** |

## Head-to-Head Benchmarks

The scores below come from **Liquid AI's common evaluation table**, which evaluated both models under the same comparison setup. Higher is better.

| Benchmark | What it tests | **Qwen3.5-0.8B Instruct** | **LFM2.5-350M** | Winner |
|---|---|---:|---:|---|
| **GPQA Diamond** | Hard graduate-level science reasoning | 27.41 | **30.64** | LFM |
| **MMLU-Pro** | General knowledge + reasoning | **37.42** | 20.01 | Qwen |
| **IFEval** | Exact instruction following | 59.94 | **76.96** | LFM |
| **IFBench** | Difficult multi-constraint instruction following | 22.87 | **40.69** | LFM |
| **Multi-IF** | Multilingual instruction following | 41.68 | **44.92** | LFM |
| **CaseReportBench** | Structured information / report reasoning | 13.83 | **32.45** | LFM |
| **BFCL v3** | Function / tool calling | 35.08 | **44.11** | LFM |
| **BFCL v4** | Harder function / tool calling | 18.70 | **21.86** | LFM |
| **τ²-Bench Telecom** | Agent/tool-use task completion | 12.57 | **18.86** | LFM |
| **τ²-Bench Retail** | Agent/tool-use task completion | 6.14 | **17.84** | LFM |

## Qwen's Own Published Scores

Qwen also publishes its own evaluation under a different setup. These numbers should **not be directly mixed with Liquid AI's table above**.

| Benchmark | Qwen3.5-0.8B Non-Thinking | Qwen3.5-0.8B Thinking |
|---|---:|---:|
| **MMLU-Pro** | 29.7 | **42.3** |
| **IFEval** | **52.1** | 44.0 |
| **IFBench** | — | 21.0 |
| **BFCL v4** | — | 25.3 |
| **LongBench v2** | — | 26.1 |

Different prompts, decoding settings, benchmark harnesses, and thinking modes can produce materially different scores. For direct model-vs-model comparison, use the common Liquid AI table above.

## Architecture Summary

### Qwen3.5-0.8B

```text
18 × Gated DeltaNet blocks   ← linear attention
 6 × Gated Attention blocks  ← normal attention
───────────────────────────
24 language-model layers
```

`Gated DeltaNet` is a **linear-attention mechanism**. It is not Mamba and not a State Space Model (SSM).

Qwen3.5-0.8B also includes a **vision encoder**, so it can consume images as well as text.

### LFM2.5-350M

```text
10 × double-gated convolution blocks
 6 × GQA attention blocks
────────────────────────────
16 layers
```

`GQA` = **Grouped Query Attention**: several query heads share key/value heads, reducing memory and compute.

LFM is not Mamba, not an SSM, and not a linear-attention model. Its unusual part is the heavy use of **short convolution blocks** between attention layers.

## Practical Choice

| Use case | Pick |
|---|---|
| **General-purpose tiny LLM** | **Qwen3.5-0.8B** |
| **Knowledge / reasoning** | **Qwen3.5-0.8B** |
| **Very long context** | **Qwen3.5-0.8B** |
| **Vision + text** | **Qwen3.5-0.8B** |
| **Smallest possible model** | **LFM2.5-350M** |
| **Structured JSON / extraction** | **LFM2.5-350M** |
| **Instruction following** | **LFM2.5-350M** |
| **Tool / function calling at tiny size** | **LFM2.5-350M** |
| **CPU / edge deployment** | **LFM2.5-350M** |

## Model Links

- Qwen3.5-0.8B: https://huggingface.co/Qwen/Qwen3.5-0.8B
- LFM2.5-350M: https://huggingface.co/LiquidAI/LFM2.5-350M

## Sources

- Qwen3.5-0.8B official model card:
  https://huggingface.co/Qwen/Qwen3.5-0.8B
- LFM2.5-350M official model card and common benchmark comparison:
  https://huggingface.co/LiquidAI/LFM2.5-350M
