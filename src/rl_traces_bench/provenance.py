"""Capture which vLLM build produced a run, for attribution in compare."""
import subprocess

def _git(path, *args):
    try:
        return subprocess.check_output(["git", "-C", path, *args],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

def git_sha(path):
    return _git(path, "rev-parse", "HEAD")

def _dirty(path):
    out = _git(path, "status", "--porcelain")
    return bool(out) if out is not None else None

def _vllm_version():
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except Exception:
        return None

def collect_provenance(vllm_src=None):
    return {"vllm_version": _vllm_version(), "vllm_src": vllm_src,
            "git_sha": git_sha(vllm_src) if vllm_src else None,
            "dirty": _dirty(vllm_src) if vllm_src else None}
