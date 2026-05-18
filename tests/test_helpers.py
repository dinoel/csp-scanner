"""Scanner utility function tests."""
import math
from datetime import date, datetime, timedelta, timezone
import pandas as pd
import pytest
from analytics import _mid_or_last, get_ma200, compute_hv30, calc_expected_move, pick_expiry
from csp_scanner import _int, _r, _effective_price, check_earnings, _last_valid_trade_date


class TestInt:
    def test_int_from_float(self):
        assert _int(5.9) == 5

    def test_nan_returns_zero(self):
        assert _int(float("nan")) == 0

    def test_none_returns_zero(self):
        assert _int(None) == 0

    def test_string_number(self):
        assert _int("42") == 42


class TestR:
    def test_rounds_to_n_places(self):
        assert _r(3.14159, 2) == 3.14
        assert _r(3.14159, 0) == 3.0

    def test_default_two_places(self):
        assert _r(1.005) == 1.0  # floating point, but default is 2

    def test_nan_passthrough(self):
        assert math.isnan(_r(float("nan")))

    def test_none_returns_nan(self):
        assert math.isnan(_r(None))


class TestMidOrLast:
    def test_mid_when_bid_and_ask(self):
        assert _mid_or_last({"bid": 1.0, "ask": 3.0}) == 2.0

    def test_bid_when_no_ask(self):
        assert _mid_or_last({"bid": 2.0, "ask": 0}) == 2.0

    def test_last_price_fallback(self):
        assert _mid_or_last({"bid": 0, "ask": 0, "lastPrice": 1.5}) == 1.5

    def test_nan_when_all_zero(self):
        assert math.isnan(_mid_or_last({"bid": 0, "ask": 0, "lastPrice": 0}))


class TestEffectivePrice:
    # Use today as lastTradeDate — always fresh
    BASE = {"strike": 100, "bid": 2.0, "ask": 2.4,
            "lastPrice": 0, "volume": 500, "openInterest": 100,
            "lastTradeDate": datetime.now(timezone.utc)}

    def test_returns_mid_and_bid_for_liquid(self):
        price_iv, bid_ret = _effective_price(self.BASE)
        assert price_iv == pytest.approx(2.2)
        assert bid_ret  == pytest.approx(2.0)

    def test_fails_oi_filter(self):
        row = {**self.BASE, "openInterest": 10}
        p, b = _effective_price(row)
        assert math.isnan(p) and math.isnan(b)

    def test_fails_volume_filter(self):
        row = {**self.BASE, "volume": 5}
        p, b = _effective_price(row)
        assert math.isnan(p) and math.isnan(b)

    def test_fails_spread_filter(self):
        row = {**self.BASE, "bid": 1.0, "ask": 10.0}
        p, b = _effective_price(row)
        assert math.isnan(p) and math.isnan(b)

    def test_fails_stale_last_trade(self):
        stale = datetime.now(timezone.utc) - timedelta(days=30)
        row = {**self.BASE, "lastTradeDate": stale}
        p, b = _effective_price(row)
        assert math.isnan(p) and math.isnan(b)

    def test_no_last_trade_date_passes(self):
        # When lastTradeDate is absent, staleness check is skipped
        row = {k: v for k, v in self.BASE.items() if k != "lastTradeDate"}
        p, b = _effective_price(row)
        assert not math.isnan(p)


class TestLastValidTradeDate:
    """Test _last_valid_trade_date with controlled 'now' values."""

    def _et(self, iso: str) -> datetime:
        """Build a timezone-aware ET datetime from 'YYYY-MM-DD HH:MM' string."""
        from zoneinfo import ZoneInfo
        return datetime.strptime(iso, "%Y-%m-%d %H:%M").replace(
            tzinfo=ZoneInfo("America/New_York")
        )

    def test_saturday_returns_friday(self):
        # 2026-05-16 is Saturday → should return Friday 2026-05-15
        result = _last_valid_trade_date(self._et("2026-05-16 12:00"))
        assert result == date(2026, 5, 15)
        assert result.weekday() == 4  # Friday

    def test_sunday_returns_friday(self):
        result = _last_valid_trade_date(self._et("2026-05-17 10:00"))
        assert result == date(2026, 5, 15)
        assert result.weekday() == 4

    def test_monday_before_grace_returns_friday(self):
        # Monday 08:00 ET → before open → accept Friday
        result = _last_valid_trade_date(self._et("2026-05-18 08:00"))
        assert result == date(2026, 5, 15)
        assert result.weekday() == 4

    def test_monday_within_grace_returns_friday(self):
        # Monday 09:45 ET → within 30-min grace → accept Friday
        result = _last_valid_trade_date(self._et("2026-05-18 09:45"))
        assert result == date(2026, 5, 15)

    def test_monday_after_grace_returns_today(self):
        # Monday 10:30 ET → past grace → require today
        result = _last_valid_trade_date(self._et("2026-05-18 10:30"))
        assert result == date(2026, 5, 18)

    def test_wednesday_before_open_returns_tuesday(self):
        result = _last_valid_trade_date(self._et("2026-05-20 07:00"))
        assert result == date(2026, 5, 19)  # Tuesday

    def test_wednesday_after_grace_returns_today(self):
        result = _last_valid_trade_date(self._et("2026-05-20 11:00"))
        assert result == date(2026, 5, 20)


class TestCheckEarnings:
    def test_earnings_in_window(self):
        cal = {"Earnings Date": ["2026-06-15"]}
        result = check_earnings(cal, date(2026, 5, 17), date(2026, 6, 20))
        assert result == "2026-06-15"

    def test_earnings_after_expiry(self):
        cal = {"Earnings Date": ["2026-07-01"]}
        result = check_earnings(cal, date(2026, 5, 17), date(2026, 6, 20))
        assert result is None

    def test_earnings_before_today(self):
        cal = {"Earnings Date": ["2026-05-10"]}
        result = check_earnings(cal, date(2026, 5, 17), date(2026, 6, 20))
        assert result is None

    def test_empty_calendar(self):
        assert check_earnings({}, date(2026, 5, 17), date(2026, 6, 20)) is None

    def test_none_calendar(self):
        assert check_earnings(None, date(2026, 5, 17), date(2026, 6, 20)) is None


class TestPickExpiry:
    TODAY = date(2026, 5, 17)

    def test_picks_within_dte_window(self):
        # 2026-05-20 → DTE=3 (< dte_min=7, excluded)
        # 2026-06-20 → DTE=34 (in window)
        # 2026-09-20 → DTE=126 (> dte_max=51, excluded)
        exps = ["2026-05-20", "2026-06-20", "2026-09-20"]
        exp, dte = pick_expiry(exps, self.TODAY, 7, 51)
        assert exp == "2026-06-20"
        assert dte == 34

    def test_returns_none_when_no_match(self):
        exp, dte = pick_expiry(["2026-09-20"], self.TODAY, 7, 51)
        assert exp is None
        assert dte is None

    def test_picks_nearest_expiry(self):
        exps = ["2026-06-05", "2026-06-20"]
        exp, dte = pick_expiry(exps, self.TODAY, 7, 51)
        assert exp == "2026-06-05"


class TestGetMa200:
    def _make_hist(self, n: int, price: float = 100.0) -> pd.DataFrame:
        return pd.DataFrame({"Close": [price] * n})

    def test_constant_price_equals_price(self):
        hist = self._make_hist(250, 150.0)
        assert get_ma200(hist) == pytest.approx(150.0)

    def test_insufficient_data_returns_nan(self):
        assert math.isnan(get_ma200(self._make_hist(100)))

    def test_none_returns_nan(self):
        assert math.isnan(get_ma200(None))


class TestComputeHv30:
    def test_zero_vol_near_zero(self):
        # Constant price → zero log returns → HV ≈ 0
        hist = pd.DataFrame({"Close": [100.0] * 60})
        hv = compute_hv30(hist)
        assert hv == pytest.approx(0.0, abs=1e-6)

    def test_insufficient_data_returns_nan(self):
        hist = pd.DataFrame({"Close": [100.0] * 20})
        assert math.isnan(compute_hv30(hist))

    def test_volatile_series_positive(self):
        import numpy as np
        rng = np.random.default_rng(42)
        prices = 100 * (1 + rng.normal(0, 0.02, 60)).cumprod()
        hist = pd.DataFrame({"Close": prices})
        assert compute_hv30(hist) > 0


class TestCalcExpectedMove:
    def _make_chain(self, strikes, bids, asks):
        return pd.DataFrame({"strike": strikes, "bid": bids,
                             "ask": asks, "lastPrice": bids})

    def test_atm_straddle(self):
        calls = self._make_chain([100], [3.0], [3.2])
        puts  = self._make_chain([100], [2.8], [3.0])
        em = calc_expected_move(calls, puts, 100)
        assert em == pytest.approx(6.0, abs=0.1)

    def test_no_shared_strikes_returns_nan(self):
        calls = self._make_chain([100], [3.0], [3.2])
        puts  = self._make_chain([110], [2.8], [3.0])
        assert math.isnan(calc_expected_move(calls, puts, 100))
