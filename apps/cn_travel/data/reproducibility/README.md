# Frozen source contract

`pipeline_sources/` contains preserved hash-addressed generator and source
snapshots referenced by the sealed Stage 2 and Stage 3 manifests.
[`../../eval/reproducibility/evaluator_sources/20260901_all_entries/`](../../eval/reproducibility/evaluator_sources/20260901_all_entries/)
contains the evaluator and channel sources named by that evaluation contract.
Python snapshots use deterministic gzip containers; their decompressed bytes
are the immutable, hash-addressed provenance artifacts and remain outside
runtime import paths.

Validation hashes each preserved snapshot against its corresponding manifest
identity and hashes all 1,010 generated records against their sealed record
maps.
