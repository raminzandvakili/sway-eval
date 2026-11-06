"""Load a Qwen-family Decision-2.0 package with a lower memory peak.

The package's own loader (``decision2/qwen.py``, ``QwenDecision.load``)
builds the model in FP32 on the CPU, converts the whole model with
``model.float()``, rounds every BF16-exact ``nn.Linear`` weight and bias of
the backbone back to BF16 (``keep_linear_bf16``) and only then moves it to
the GPU. For a 27B model the FP32 copy alone is about 117 GB, more than a
121 GB machine whose CPU and GPU share memory can hold.

This loader runs the same package code with these changes, none of which
alters a value:

- weights load straight to the GPU, each one cloned out of the file's memory
  map first (as ``_memory.cloning_before_copy`` does for Clef);
- a BF16 tensor that the package asks for in FP32 stays BF16 while loading;
  afterwards every tensor that the package's rule would not hold in BF16 is
  converted to FP32 one at a time. BF16 to FP32 is exact, and the package
  rounds an FP32 value back to BF16 only when the rounding is exact, so each
  tensor ends with the same dtype and bytes as on the official path;
- a LoRA adapter's layers are built empty by peft and each adapter weight
  is assigned from the file, as stored (FP32), straight to the GPU. The
  official path copies the file's values into FP32 parameters; letting peft
  load them here would cast them to the BF16 dtype of the base layers;
- checkpoint files are read with safetensors' ``pread`` backend instead of
  being memory-mapped, so pages already copied can be dropped from the page
  cache while loading (``_memory.evicting``); the bytes read are the same;
- transformers' allocator warm-up, which reserves GPU memory for the whole
  model in the requested dtype (FP32) before loading, is skipped. It only
  speeds loading up.

Only the ``qwen-full`` and ``qwen-adapter`` profiles without a storage codec,
residual or label-token readout are supported; anything else raises.
Equivalence is checked against the official loader where it fits
(``experiments/checks/loader_equivalence.py`` in the study repo).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def _patched(target: str):
    import torch
    from peft import PeftModel
    from transformers import PreTrainedModel
    from transformers import core_model_loading as cml
    from transformers import modeling_utils as mu

    opener, warmup, material, pretrained, peft_pretrained = mu.safe_open, mu.caching_allocator_warmup, cml._materialize_copy, PreTrainedModel.from_pretrained, PeftModel.from_pretrained

    def materialize(tensor, device=None, dtype=None):
        tensor = tensor[...]
        if device is not None:
            tensor = tensor.clone()
        if dtype == torch.float32 and tensor.dtype == torch.bfloat16:
            dtype = None  # converted later, only where the package's rule keeps FP32
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("device_map", {"": target})
        return pretrained.__func__(cls, *args, **kwargs)

    def peft_from_pretrained(cls, model, model_id, adapter_name="default", is_trainable=False, **kwargs):
        return _lora_from_files(cls, model, Path(model_id), adapter_name, is_trainable, target, torch)

    def safe_open(*args, **kwargs):
        if kwargs.get("backend") == "mmap":
            kwargs["backend"] = "pread"
        return opener(*args, **kwargs)

    mu.safe_open = safe_open
    mu.caching_allocator_warmup = lambda *args, **kwargs: None
    cml._materialize_copy = materialize
    PreTrainedModel.from_pretrained = classmethod(from_pretrained)
    PeftModel.from_pretrained = classmethod(peft_from_pretrained)
    try:
        yield
    finally:
        mu.safe_open = opener
        mu.caching_allocator_warmup = warmup
        cml._materialize_copy = material
        PreTrainedModel.from_pretrained = pretrained
        PeftModel.from_pretrained = peft_pretrained


def _lora_from_files(cls: Any, model: Any, path: Path, adapter_name: str, is_trainable: bool, target: str, torch: Any) -> Any:
    """``PeftModel.from_pretrained`` for a local LoRA adapter, with every
    adapter weight taken from the file unchanged and placed on ``target``."""
    from peft import LoraConfig
    from safetensors import safe_open

    config = LoraConfig.from_pretrained(str(path))
    if config.task_type is not None:
        raise NotImplementedError("low-peak loading covers LoRA adapters without a task type")
    config.inference_mode = not is_trainable
    peft_model = cls(model, config, adapter_name, low_cpu_mem_usage=True)
    params = dict(peft_model.named_parameters())
    wanted = {n for n in params if ".lora_" in n}
    with safe_open(path / "adapter_model.safetensors", "pt", device=target) as f:
        for key in f.keys():
            name = key.replace(".lora_A.", f".lora_A.{adapter_name}.").replace(".lora_B.", f".lora_B.{adapter_name}.")
            if name not in wanted:
                raise ValueError(f"adapter tensor {key} has no LoRA parameter")
            module_name, _, attr = name.rpartition(".")
            module = peft_model.get_submodule(module_name)
            if list(module._parameters[attr].shape) != list(f.get_slice(key).get_shape()):
                raise ValueError(f"adapter tensor {key} has the wrong shape")
            module._parameters[attr] = torch.nn.Parameter(f.get_tensor(key), requires_grad=is_trainable)
            wanted.discard(name)
    if wanted:
        raise ValueError(f"{len(wanted)} LoRA parameters have no adapter tensor, e.g. {sorted(wanted)[0]}")
    if not is_trainable:
        peft_model.eval()
    return peft_model


def to_package_dtypes(model: Any, torch: Any) -> None:
    """Convert to FP32, one tensor at a time, every floating tensor the package
    would not hold in BF16: all but the unshared ``nn.Linear`` weights and
    biases of the backbone (``keep_linear_bf16`` rounds those afterwards)."""
    backbone = model.backbone
    shared = {id(p) for m in backbone.modules() if not isinstance(m, torch.nn.Linear) for p in m.parameters(recurse=False)}
    keep = {id(p) for m in backbone.modules() if isinstance(m, torch.nn.Linear)
            for p in (m.weight, m.bias) if p is not None and id(p) not in shared}
    for m in model.modules():
        for name, p in list(m._parameters.items()):
            if p is not None and p.is_floating_point() and p.dtype != torch.float32 and id(p) not in keep:
                p.data = p.data.float()
        for name, b in list(m._buffers.items()):
            if b is not None and b.is_floating_point() and b.dtype != torch.float32:
                m._buffers[name] = b.float()


def load_low_peak(path: str | Path, device: str = "cuda:0") -> Any:
    """A ``decision2.Decision2`` built like ``Decision2.from_pretrained(path,
    device=device, graphs=False, kernels=False, share_context=False)``."""
    import torch
    from decision2 import qwen as q
    from decision2.api import Decision2, verify_bundle
    from decision2._vendor.dev2model.calibration import load_calibration
    from decision2._vendor.dev2model.decision_model import DecisionModel
    from decision2._vendor.dev2model.infer import checkpoint_fingerprint

    manifest = verify_bundle(path)
    root = Path(path).resolve(strict=True)
    if manifest["profile"] not in ("qwen-full", "qwen-adapter") or manifest.get("storage"):
        raise NotImplementedError("low-peak loading covers qwen-full and qwen-adapter packages without a storage codec")
    metadata = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    if metadata.get("dec_residual") is not None or metadata.get("readout") == "label_token":
        raise NotImplementedError("low-peak loading does not cover residual or label-token packages")
    source = q.resolve_base(manifest["base"], None) if manifest["profile"] == "qwen-adapter" else None
    identity = checkpoint_fingerprint(root, source)
    if identity["model_sha256"] != manifest["identity"]["model_sha256"]:
        raise ValueError("Model identity differs from the scored checkpoint")
    score_bias = q.load_score_bias_entry(root, manifest, identity["model_sha256"])
    calibration = manifest.get("calibration")
    if calibration:
        temperatures, report = load_calibration(root / calibration["file"], identity["model_sha256"])
        if report.get("inference", {}).get("max_length") not in (None, manifest["max_input_tokens"]):
            raise ValueError("Calibration context differs from the package limit")
    else:
        temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("low-peak loading needs a CUDA BF16 GPU")
    with _patched(str(target)):
        model, tokenizer = DecisionModel.from_checkpoint(root, source_path=source)
    torch.cuda.empty_cache()
    model = model.to(target)  # the head is built on the CPU
    to_package_dtypes(model, torch)
    residency = q.keep_linear_bf16(model.backbone, torch)
    torch.cuda.empty_cache()  # return the BF16 copies replaced above
    model = model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")
    from decision2.fast import install

    fast = install(model, torch, graphs=False, kernels=False)
    backend = q.QwenDecision(model, tokenizer, target, temperatures, manifest["max_input_tokens"], torch, score_bias,
                             residency, None, q.forward_token_budget(model.backbone.config), fast, False)
    loaded = backend.parameter_count()
    if loaded != manifest["parameters"]["loaded"]:
        raise ValueError(f"Loaded {loaded:,} parameters; the manifest declares {manifest['parameters']['loaded']:,}")
    return Decision2(backend, manifest, root)
