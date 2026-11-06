"""Keep model files out of the page cache while a local model loads.

On machines where the CPU and GPU share memory (such as a GB10), CUDA
allocations come out of free memory and do not reclaim the page cache, so
reading tens of gigabytes of weights can make a load fail with memory to
spare. While loading, a background thread asks the kernel to drop the cached
pages of the files under ``root`` (``posix_fadvise(DONTNEED)``, no root
needed). Weight values are unaffected.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path


def evict(root: str | Path) -> int:
    """Drop the cached pages of every regular file under ``root``; returns how many."""
    n = 0
    for d, _, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    return n


@contextmanager
def evicting(root: str | Path | None, every_s: float = 2.0):
    """Evict ``root`` before, during (every ``every_s`` seconds) and after the block."""
    if root is None:
        yield
        return
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(every_s):
            evict(root)

    evict(root)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        evict(root)


@contextmanager
def cloning_before_copy(enabled: bool = True):
    """While loading with transformers, copy each weight out of the file's
    memory map before moving it to the GPU.

    On a GPU that shares memory with the CPU, moving a tensor straight from a
    private memory map to the device duplicates its pages in host memory until
    loading ends, so a model of size M needs about 2M. Cloning first keeps the
    peak near M. The clone is exact, so weight values are unchanged. Files
    are also read with safetensors' ``pread`` backend instead of being
    memory-mapped, so pages already copied can be dropped from the page cache
    while loading (``evicting``); the bytes read are the same.
    Patches ``transformers.core_model_loading._materialize_copy`` and the
    ``safe_open`` used by ``transformers.modeling_utils`` for the block only.
    """
    if not enabled:
        yield
        return
    from transformers import core_model_loading as cml
    from transformers import modeling_utils as mu

    original, opener = cml._materialize_copy, mu.safe_open

    def safe_open(*args, **kwargs):
        if kwargs.get("backend", "mmap") == "mmap":
            kwargs["backend"] = "pread"
        return opener(*args, **kwargs)

    def materialize(tensor, device=None, dtype=None):
        tensor = tensor[...]
        if device is not None:
            tensor = tensor.clone()
        if dtype is not None or device is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    cml._materialize_copy, mu.safe_open = materialize, safe_open
    try:
        yield
    finally:
        cml._materialize_copy, mu.safe_open = original, opener
