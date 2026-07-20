import urllib.error

from rl_traces_bench import doctor
from rl_traces_bench.doctor import run_checks


def test_always_present_checks_with_no_url_and_no_serve_args():
    checks = run_checks({"TOKENIZER": "org/m"})
    names = {c[0] for c in checks}
    assert names == {"aiperf present", "tokenizer set"}
    assert all(len(c) == 3 for c in checks)


def test_client_only_env_gets_endpoint_check_not_vllm_checks(monkeypatch):
    monkeypatch.setattr(doctor, "_endpoint_reachable", lambda url: True)
    checks = run_checks({"URL": "localhost:8000", "TOKENIZER": "org/m"})
    names = {c[0] for c in checks}
    assert "endpoint reachable" in names
    assert "vllm importable" not in names
    assert "vllm_src exists" not in names


def test_serving_env_gets_vllm_checks(monkeypatch):
    monkeypatch.setattr(doctor, "_importable", lambda mod: True)
    checks = run_checks({
        "VLLM_SERVE_ARGS": "org/model --port 8000",
        "VLLM_SRC": "/nope/vllm",
        "TOKENIZER": "org/m",
    })
    names = {c[0] for c in checks}
    assert {"vllm importable", "vllm_src exists"} <= names


def test_aiperf_absent_fails_check(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    checks = run_checks({"URL": "localhost:8000", "TOKENIZER": "org/m"})
    d = {c[0]: c[1] for c in checks}
    assert d["aiperf present"] is False
    assert "vllm importable" not in d


def test_vllm_src_exists_check_reflects_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "_importable", lambda mod: True)
    checks = run_checks({
        "VLLM_SERVE_ARGS": "org/model --port 8000",
        "VLLM_SRC": str(tmp_path),
        "TOKENIZER": "org/m",
    })
    d = {c[0]: c[1] for c in checks}
    assert d["vllm_src exists"] is True

    checks2 = run_checks({
        "VLLM_SERVE_ARGS": "org/model --port 8000",
        "VLLM_SRC": "/definitely/not/a/real/path",
        "TOKENIZER": "org/m",
    })
    d2 = {c[0]: c[1] for c in checks2}
    assert d2["vllm_src exists"] is False


def test_endpoint_reachable_true_on_success(monkeypatch):
    class FakeResp:
        pass
    monkeypatch.setattr(doctor.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    assert doctor._endpoint_reachable("localhost:8000") is True


def test_endpoint_reachable_true_on_http_error(monkeypatch):
    def raise_http_error(*a, **k):
        raise urllib.error.HTTPError("url", 404, "not found", {}, None)
    monkeypatch.setattr(doctor.urllib.request, "urlopen", raise_http_error)
    assert doctor._endpoint_reachable("localhost:8000") is True


def test_endpoint_reachable_false_on_url_error(monkeypatch):
    def raise_url_error(*a, **k):
        raise urllib.error.URLError("boom")
    monkeypatch.setattr(doctor.urllib.request, "urlopen", raise_url_error)
    assert doctor._endpoint_reachable("localhost:8000") is False


def test_endpoint_reachable_strips_existing_scheme_and_targets_v1_models(monkeypatch):
    seen = {}
    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        class R:
            pass
        return R()
    monkeypatch.setattr(doctor.urllib.request, "urlopen", fake_urlopen)
    doctor._endpoint_reachable("http://localhost:8000/")
    assert seen["url"] == "http://localhost:8000/v1/models"


def test_run_checks_boolean_outcomes_end_to_end(monkeypatch):
    # aiperf missing, tokenizer unset, URL unreachable -> all fail; no vllm checks (no serve args)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    monkeypatch.setattr(doctor, "_endpoint_reachable", lambda url: False)
    checks = run_checks({"URL": "localhost:8000"})
    d = {c[0]: c[1] for c in checks}
    assert d == {"aiperf present": False, "tokenizer set": False, "endpoint reachable": False}
