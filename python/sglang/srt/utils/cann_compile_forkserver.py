# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Keep the NPU memory-saver preload out of CANN's CPU compiler workers."""

import multiprocessing
import os
import re


def prepare_cann_compile_forkserver():
    """Start CANN's forkserver before model compilation, preserving parent TMS.

    CANN explicitly uses forkserver with an empty preload-module list. Loading
    the TMS native library in that fresh interpreter leaves SIGINT represented
    as None; CPython 3.12 cannot restore it in forkserver children. Only the CPU
    compiler server omits TMS. The scheduler retains its loaded allocator and
    original environment, and later CANN workers reuse the clean server.
    """
    preload = os.environ.get("LD_PRELOAD", "")
    libraries = re.split(r"[:\s]+", preload.strip())
    remaining = [
        lib for lib in libraries
        if lib and "torch_memory_saver_hook_mode_preload" not in os.path.basename(lib)
    ]
    if len(remaining) == len([lib for lib in libraries if lib]):
        return False

    context = multiprocessing.get_context("forkserver")
    context.set_forkserver_preload([])
    try:
        if remaining:
            os.environ["LD_PRELOAD"] = ":".join(remaining)
        else:
            os.environ.pop("LD_PRELOAD", None)
        # A built-in CPU target is picklable without importing the model or NPU.
        child = context.Process(target=os.getpid)
        child.start()
        child.join(timeout=15)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
            raise RuntimeError("CANN compiler forkserver startup timed out")
        if child.exitcode != 0:
            raise RuntimeError(
                f"CANN compiler forkserver child exited with {child.exitcode}"
            )
        child.close()
    finally:
        os.environ["LD_PRELOAD"] = preload
    return True
