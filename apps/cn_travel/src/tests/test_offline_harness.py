"""Checks for deterministic offline test selection."""

import pytest

import conftest


class _OfflineConfig:
    def getoption(self, name):
        assert name == "markexpr"
        return "not live and not model and not policy and not reviews and not rag and not amap"

    def addinivalue_line(self, _name, _value):
        pass


def test_offline_configuration_never_probes_capabilities(monkeypatch):
    def unexpected_probe():
        raise AssertionError("offline collection attempted a capability probe")

    for name in ("_has_key", "_has_rag", "_has_net", "_has_model", "_has_policy",
                 "_has_reviews",
                 "_hotel_backend_is_real"):
        monkeypatch.setattr(conftest, name, unexpected_probe)

    conftest.CAPS.clear()
    conftest.pytest_configure(_OfflineConfig())

    assert conftest.CAPS == {
        "live": False,
        "rag": False,
        "amap": False,
        "model": False,
        "policy": False,
        "reviews": False,
        "hotel_backend_real": False,
        "offline": True,
    }


def test_required_capabilities_fail_closed(monkeypatch):
    monkeypatch.setenv("CN_TRAVEL_REQUIRED_CAPABILITIES", "rag,policy")
    conftest.CAPS.clear()
    conftest.CAPS.update(rag=True, policy=False)
    with pytest.raises(pytest.UsageError, match="policy"):
        conftest._require_capabilities()
