from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_ext = None
_tried = False

HERE = Path(__file__).parent


def _cupti_dirs() -> tuple[str | None, str | None]:
    """Header and lib dirs for the CUPTI torch links, so we never load a second copy."""
    try:
        import torch
    except ImportError:
        return None, None
    site = Path(torch.__file__).parent.parent
    for sub in ("nvidia/cuda_cupti", "nvidia/cu13"):
        d = site / sub
        if (d / "lib").is_dir():
            inc = d / "include" if (d / "include" / "cupti_callbacks.h").exists() else None
            return (str(inc) if inc else None), str(d / "lib")
    return None, None


def _cuda_include() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME

    for root in [CUDA_HOME, os.environ.get("CUDA_HOME"), "/usr/local/cuda"]:
        if root and (Path(root) / "include" / "cuda.h").exists():
            return str(Path(root) / "include")
    return None


def load(verbose: bool = False):
    global _ext, _tried
    if _ext is not None or _tried:
        return _ext
    _tried = True
    try:
        from torch.utils.cpp_extension import load as build
    except ImportError:
        return None
    cupti_inc, cupti_lib = _cupti_dirs()
    cuda_inc = _cuda_include()
    if cupti_lib is None or cuda_inc is None:
        warnings.warn(
            "vramxray: native mode needs the CUDA headers and torch's CUPTI wheel; "
            "staying in pure mode",
            stacklevel=2,
        )
        return None
    includes = [cuda_inc]
    if cupti_inc:
        includes.append(cupti_inc)
    else:
        extras = Path(cuda_inc).parent / "extras" / "CUPTI" / "include"
        if extras.is_dir():
            includes.append(str(extras))
    soname = next((f for f in os.listdir(cupti_lib) if f.startswith("libcupti.so.")), None)
    if soname is None:
        return None
    try:
        _ext = build(
            name="vramxray_core",
            sources=sorted(str(p) for p in HERE.glob("*.cpp")),
            extra_include_paths=[str(HERE), *includes],
            extra_ldflags=[
                "-lc10_cuda",
                f"-L{cupti_lib}",
                f"-l:{soname}",
                f"-Wl,-rpath,{cupti_lib}",
                "-ldl",
            ],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=verbose,
        )
    except Exception as e:  # build failures are expected on machines without a toolchain
        warnings.warn(
            f"vramxray: native build failed, staying in pure mode ({e!s:.200})", stacklevel=2
        )
        print(f"vramxray native build error: {e}", file=sys.stderr) if verbose else None
        return None
    return _ext
