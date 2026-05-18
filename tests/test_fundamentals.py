"""Fundamental analysis logic tests."""
import math
import pytest
from fundamentals import (
    _rev_growth, _beat_rate, _fcf_margin, _debt_to_equity,
    _score_rev_growth, _score_beat_rate, _score_fcf_margin, _score_debt_equity,
    compute, FundamentalMetrics,
)
from ai import _fmt_billions

# ── Fixtures ──────────────────────────────────────────────────────────────────

INCOME_2Y = [
    {"revenue": 100e9, "operating_income": 25e9, "net_income": 20e9, "eps_diluted": 5.0},
    {"revenue":  80e9, "operating_income": 18e9, "net_income": 14e9, "eps_diluted": 3.50},
]

EARNINGS_4Q = [
    {"period": "Q4", "actual_eps": 1.50, "estimated_eps": 1.40},  # beat  +7.1%
    {"period": "Q3", "actual_eps": 1.20, "estimated_eps": 1.30},  # miss  -7.7%
    {"period": "Q2", "actual_eps": 2.00, "estimated_eps": 1.90},  # beat  +5.3%
    {"period": "Q1", "actual_eps": 0.80, "estimated_eps": 0.80},  # tie  → beat (>=)
]

BALANCE_1 = [
    {"total_debt": 20e9, "total_equity": 50e9, "cash_and_equivalents": 30e9}
]

CASHFLOW_1 = [
    {"operating_cash_flow": 30e9, "capital_expenditure": -5e9, "free_cash_flow": 25e9}
]


# ── Revenue growth ────────────────────────────────────────────────────────────

class TestRevGrowth:
    def test_positive_growth(self):
        growth = _rev_growth(INCOME_2Y)
        assert abs(growth - 25.0) < 0.01  # (100-80)/80 = 25%

    def test_negative_growth(self):
        income = [{"revenue": 80e9}, {"revenue": 100e9}]
        assert _rev_growth(income) < 0

    def test_single_period_returns_nan(self):
        assert math.isnan(_rev_growth([INCOME_2Y[0]]))

    def test_empty_returns_nan(self):
        assert math.isnan(_rev_growth([]))

    def test_zero_prev_revenue_returns_nan(self):
        income = [{"revenue": 100e9}, {"revenue": 0}]
        assert math.isnan(_rev_growth(income))

    def test_fallback_field_name(self):
        income = [{"total_revenue": 120e9}, {"total_revenue": 100e9}]
        assert abs(_rev_growth(income) - 20.0) < 0.01


# ── EPS beat rate ─────────────────────────────────────────────────────────────

class TestBeatRate:
    def test_three_of_four(self):
        assert _beat_rate(EARNINGS_4Q) == pytest.approx(75.0)

    def test_all_beat(self):
        earnings = [{"actual_eps": 2.0, "estimated_eps": 1.0}] * 8
        assert _beat_rate(earnings) == pytest.approx(100.0)

    def test_all_miss(self):
        earnings = [{"actual_eps": 0.5, "estimated_eps": 1.0}] * 8
        assert _beat_rate(earnings) == pytest.approx(0.0)

    def test_empty_returns_nan(self):
        assert math.isnan(_beat_rate([]))

    def test_missing_estimate_skipped(self):
        earnings = [
            {"actual_eps": 1.0, "estimated_eps": 0.9},   # beat
            {"actual_eps": 1.0},                          # no estimate → skip
        ]
        assert _beat_rate(earnings) == pytest.approx(100.0)


# ── FCF margin ────────────────────────────────────────────────────────────────

class TestFcfMargin:
    def test_from_explicit_fcf_field(self):
        margin = _fcf_margin(INCOME_2Y, CASHFLOW_1)
        assert abs(margin - 25.0) < 0.01  # 25B / 100B

    def test_computed_from_op_cf_minus_capex(self):
        cf = [{"operating_cash_flow": 30e9, "capital_expenditure": -5e9}]
        margin = _fcf_margin(INCOME_2Y, cf)
        assert abs(margin - 25.0) < 0.01

    def test_negative_fcf(self):
        cf = [{"free_cash_flow": -5e9}]
        margin = _fcf_margin(INCOME_2Y, cf)
        assert margin < 0

    def test_empty_cashflow_returns_nan(self):
        assert math.isnan(_fcf_margin(INCOME_2Y, []))

    def test_empty_income_returns_nan(self):
        assert math.isnan(_fcf_margin([], CASHFLOW_1))


# ── Debt-to-equity ────────────────────────────────────────────────────────────

class TestDebtToEquity:
    def test_basic_ratio(self):
        de = _debt_to_equity(BALANCE_1)
        assert abs(de - 0.4) < 0.001  # 20B / 50B

    def test_high_leverage(self):
        balance = [{"total_debt": 200e9, "total_equity": 50e9}]
        assert _debt_to_equity(balance) == pytest.approx(4.0)

    def test_empty_returns_nan(self):
        assert math.isnan(_debt_to_equity([]))

    def test_zero_equity_returns_nan(self):
        balance = [{"total_debt": 10e9, "total_equity": 0}]
        assert math.isnan(_debt_to_equity(balance))

    def test_fallback_field_names(self):
        balance = [{"long_term_debt": 20e9, "shareholders_equity": 50e9}]
        assert abs(_debt_to_equity(balance) - 0.4) < 0.001


# ── Sub-scores ────────────────────────────────────────────────────────────────

class TestSubScores:
    @pytest.mark.parametrize("growth,expected", [
        (30,  95), (20, 80), (10, 65), (2, 45), (-5, 25), (-15, 5),
    ])
    def test_rev_growth_buckets(self, growth, expected):
        assert _score_rev_growth(growth) == expected

    @pytest.mark.parametrize("rate,expected", [
        (90, 95), (80, 80), (70, 60), (55, 35), (40, 10),
    ])
    def test_beat_rate_buckets(self, rate, expected):
        assert _score_beat_rate(rate) == expected

    @pytest.mark.parametrize("margin,expected", [
        (25, 95), (15, 75), (7, 55), (2, 30), (-5, 5),
    ])
    def test_fcf_margin_buckets(self, margin, expected):
        assert _score_fcf_margin(margin) == expected

    @pytest.mark.parametrize("ratio,lo,hi", [
        (-0.5, 85, 100),  # net cash
        (0.3,  80, 95),   # very low debt
        (0.8,  60, 80),   # low debt
        (1.5,  40, 65),   # moderate
        (2.5,  15, 35),   # high debt
        (4.0,  0,  10),   # extreme leverage
    ])
    def test_debt_equity_buckets(self, ratio, lo, hi):
        score = _score_debt_equity(ratio)
        assert lo <= score <= hi

    def test_nan_inputs_return_neutral(self):
        nan = float("nan")
        assert _score_rev_growth(nan) == 50.0
        assert _score_beat_rate(nan)  == 50.0
        assert _score_fcf_margin(nan) == 50.0
        assert _score_debt_equity(nan)== 50.0


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestCompute:
    def test_returns_metrics_dataclass(self):
        m = compute(INCOME_2Y, CASHFLOW_1, BALANCE_1, EARNINGS_4Q)
        assert isinstance(m, FundamentalMetrics)

    def test_score_in_range(self):
        m = compute(INCOME_2Y, CASHFLOW_1, BALANCE_1, EARNINGS_4Q)
        assert 0 <= m.score <= 100

    def test_good_company_scores_high(self):
        # All top-tier metrics → score should be well above 50
        m = compute(INCOME_2Y, CASHFLOW_1, BALANCE_1, EARNINGS_4Q)
        assert m.score > 60

    def test_empty_data_gives_neutral_score(self):
        # All metrics fall back to nan → all sub-scores = 50 → composite = 50
        m = compute([], [], [], [])
        assert m.score == pytest.approx(50.0)

    def test_known_metrics_populated(self):
        m = compute(INCOME_2Y, CASHFLOW_1, BALANCE_1, EARNINGS_4Q)
        assert abs(m.rev_growth_pct - 25.0) < 0.1
        assert m.eps_beat_rate == pytest.approx(75.0)
        assert abs(m.fcf_margin_pct - 25.0) < 0.1
        assert abs(m.debt_to_equity - 0.4) < 0.01


# ── Formatting helper ─────────────────────────────────────────────────────────

class TestFmtBillions:
    def test_billions(self):
        assert _fmt_billions(2.5e9) == "$2.5B"

    def test_millions(self):
        assert _fmt_billions(500e6) == "$500.0M"

    def test_negative(self):
        assert _fmt_billions(-3e9) == "$-3.0B"

    def test_nan_returns_na(self):
        assert _fmt_billions(float("nan")) == "n/a"

    def test_none_returns_na(self):
        assert _fmt_billions(None) == "n/a"
