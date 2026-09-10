"""Pinned VERL/vLLM compatibility fixes for CN Travel trajectory GRPO.

VERL 0.9.0's in-memory LoRA loader passes the model's full HF-to-vLLM
weight mapper to ``LoRAModel.from_lora_tensors``. For LFM2 that mapper
prematurely fuses q/k/v and w1/w3 names, so vLLM receives a single tensor for
a packed module that requires a list of tensors. vLLM 0.28's normal on-disk
LoRA loader avoids this by using ``get_unstacked_mapper()``.

The launcher loads this module in every Ray and vLLM process through
``VERL_USE_EXTERNAL_MODULES``. The shim changes only VERL's tensor-backed
adapter load and is pinned fail-closed to the audited runtime versions.
"""

from __future__ import annotations

import inspect
from typing import Any

import verl
import vllm
from vllm.lora.lora_model import LoRAModel


_EXPECTED_VERL = "0.9.0"
_EXPECTED_VLLM_PREFIX = "0.28.0"


def _require_pinned_runtime() -> None:
    if verl.__version__ != _EXPECTED_VERL:
        raise RuntimeError(
            "CN Travel LoRA compatibility shim requires "
            f"verl=={_EXPECTED_VERL}, found {verl.__version__}"
        )
    if not vllm.__version__.startswith(_EXPECTED_VLLM_PREFIX):
        raise RuntimeError(
            "CN Travel LoRA compatibility shim requires vllm==0.28.0, "
            f"found {vllm.__version__}"
        )


_require_pinned_runtime()
_current_from_tensors = LoRAModel.from_lora_tensors.__func__
if not getattr(_current_from_tensors, "_cn_travel_unstacked_mapper", False):
    _original_from_tensors = _current_from_tensors
    _signature = inspect.signature(_original_from_tensors)
    if "weights_mapper" not in _signature.parameters:
        raise RuntimeError(
            "installed vLLM LoRAModel.from_lora_tensors lacks weights_mapper; "
            "refusing an unrecognized adapter-loading path"
        )

    @classmethod
    def _from_lora_tensors_unstacked(cls: type[LoRAModel], *args: Any, **kwargs: Any):
        bound = _signature.bind(cls, *args, **kwargs)
        mapper = bound.arguments.get("weights_mapper")
        if mapper is not None:
            get_unstacked_mapper = getattr(mapper, "get_unstacked_mapper", None)
            if not callable(get_unstacked_mapper):
                raise RuntimeError(
                    "LoRA weights mapper lacks get_unstacked_mapper(); "
                    "refusing an unsafe in-memory adapter load"
                )
            bound.arguments["weights_mapper"] = get_unstacked_mapper()
        return _original_from_tensors(*bound.args, **bound.kwargs)

    _from_lora_tensors_unstacked.__func__._cn_travel_unstacked_mapper = True  # type: ignore[attr-defined]
    LoRAModel.from_lora_tensors = _from_lora_tensors_unstacked


PATCH_ACTIVE = True
