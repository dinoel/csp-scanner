"""DataProvider factory and interface tests."""
import pytest
from providers import get_provider
from providers.base import DataProvider
from providers.yfinance_provider import YFinanceProvider
from providers.massive_provider import MassiveProvider


class TestGetProvider:
    def test_yfinance_returns_correct_type(self):
        p = get_provider("yfinance")
        assert isinstance(p, YFinanceProvider)

    def test_yfinance_satisfies_interface(self):
        p = get_provider("yfinance")
        assert isinstance(p, DataProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown DATA_PROVIDER"):
            get_provider("nonexistent")

    def test_massive_without_key_raises(self, monkeypatch):
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
            get_provider("massive")

    def test_massive_with_key_satisfies_interface(self, monkeypatch):
        monkeypatch.setenv("MASSIVE_API_KEY", "test_key_123")
        p = get_provider("massive")
        assert isinstance(p, DataProvider)


class TestDataProviderInterface:
    """Verify DataProvider ABC enforces all required methods."""

    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            DataProvider()  # type: ignore

    def test_incomplete_subclass_cannot_instantiate(self):
        class Incomplete(DataProvider):
            def get_spot_and_expirations(self, symbol): ...
            # missing 4 other methods

        with pytest.raises(TypeError):
            Incomplete()

    def test_complete_subclass_instantiates(self):
        class Complete(DataProvider):
            def get_spot_and_expirations(self, symbol): return {}
            def get_option_chain(self, symbol, expiry): return {}
            def get_history(self, symbol): return None
            def get_calendar(self, symbol): return {}
            def get_analyst_info(self, symbol): return {}

        p = Complete()
        assert isinstance(p, DataProvider)


class TestMassiveProviderInit:
    def test_empty_key_raises(self):
        with pytest.raises(ValueError):
            MassiveProvider(api_key="")

    def test_valid_key_sets_auth_header(self):
        p = MassiveProvider(api_key="mykey")
        assert p._session.headers["Authorization"] == "Bearer mykey"
