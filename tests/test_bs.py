"""Black-Scholes math tests."""
import math
import pytest
import pandas as pd
import numpy as np
from analytics import (
    bs_put_delta, bs_put_price, bs_put_theta,
    compute_iv, calc_profit_prob, compute_rsi,
)


class TestBsPutDelta:
    def test_atm_near_minus_half(self):
        # With r=0.05, T=1 the drift shifts delta away from -0.5 (d1≈0.35 → delta≈-0.36)
        # Use zero rate and short T to get closer to the textbook -0.5
        delta = bs_put_delta(100, 100, 0.01, 0.0, 0.20)
        assert -0.55 < delta < -0.45

    def test_deep_itm_near_minus_one(self):
        delta = bs_put_delta(100, 200, 1.0, 0.05, 0.20)
        assert delta < -0.95

    def test_deep_otm_near_zero(self):
        delta = bs_put_delta(100, 10, 1.0, 0.05, 0.20)
        assert delta > -0.05

    def test_invalid_inputs_return_nan(self):
        assert math.isnan(bs_put_delta(0,   100, 1.0, 0.05, 0.20))
        assert math.isnan(bs_put_delta(100, 0,   1.0, 0.05, 0.20))
        assert math.isnan(bs_put_delta(100, 100, 0,   0.05, 0.20))
        assert math.isnan(bs_put_delta(100, 100, 1.0, 0.05, 0.0))


class TestBsPutPrice:
    def test_positive_for_valid_inputs(self):
        price = bs_put_price(100, 100, 0.25, 0.05, 0.20)
        assert price > 0

    def test_below_strike(self):
        price = bs_put_price(100, 110, 0.25, 0.05, 0.20)
        assert price < 110

    def test_otm_cheaper_than_itm(self):
        itm = bs_put_price(100, 120, 0.25, 0.05, 0.20)
        otm = bs_put_price(100,  80, 0.25, 0.05, 0.20)
        assert itm > otm

    def test_higher_vol_means_higher_price(self):
        lo = bs_put_price(100, 100, 0.25, 0.05, 0.10)
        hi = bs_put_price(100, 100, 0.25, 0.05, 0.50)
        assert hi > lo

    def test_invalid_inputs_return_nan(self):
        assert math.isnan(bs_put_price(0, 100, 1.0, 0.05, 0.20))
        assert math.isnan(bs_put_price(100, 100, 0, 0.05, 0.20))


class TestComputeIv:
    def test_roundtrip(self):
        S, K, T, r, sigma = 100, 95, 0.25, 0.05, 0.30
        price = bs_put_price(S, K, T, r, sigma)
        recovered = compute_iv(S, K, T, r, price)
        assert abs(recovered - sigma) < 0.001

    def test_roundtrip_high_vol(self):
        S, K, T, r, sigma = 50, 45, 0.5, 0.03, 0.80
        price = bs_put_price(S, K, T, r, sigma)
        recovered = compute_iv(S, K, T, r, price)
        assert abs(recovered - sigma) < 0.005

    def test_price_below_intrinsic_returns_nan(self):
        # Put worth less than intrinsic (K-S) is arbitrage — IV undefined
        assert math.isnan(compute_iv(80, 100, 0.1, 0.05, 0.0))

    def test_zero_price_returns_nan(self):
        assert math.isnan(compute_iv(100, 95, 0.25, 0.05, 0.0))


class TestCalcProfitProb:
    def test_atm_roughly_fifty_percent(self):
        pp = calc_profit_prob(100, 100, 1.0, 0.05, 0.20)
        assert 50 < pp < 65

    def test_deep_otm_high_probability(self):
        pp = calc_profit_prob(100, 50, 1.0, 0.05, 0.20)
        assert pp > 95

    def test_deep_itm_low_probability(self):
        pp = calc_profit_prob(100, 150, 1.0, 0.05, 0.20)
        assert pp < 20

    def test_invalid_inputs_return_nan(self):
        assert math.isnan(calc_profit_prob(100, 100, 0, 0.05, 0.20))
        assert math.isnan(calc_profit_prob(100, 100, 1.0, 0.05, 0))


class TestComputeRsi:
    def _hist(self, prices):
        return pd.DataFrame({"Close": prices})

    def test_uptrend_gives_high_rsi(self):
        prices = [100 + i for i in range(50)]  # constant up
        rsi = compute_rsi(self._hist(prices))
        assert rsi > 70

    def test_downtrend_gives_low_rsi(self):
        prices = [100 - i * 0.5 for i in range(50)]  # constant down
        rsi = compute_rsi(self._hist(prices))
        assert rsi < 30

    def test_flat_gives_nan_or_fifty(self):
        prices = [100.0] * 30
        rsi = compute_rsi(self._hist(prices))
        # constant price → zero gains and losses → RS undefined → NaN or ~50
        assert math.isnan(rsi) or 40 < rsi < 60

    def test_insufficient_data_returns_nan(self):
        assert math.isnan(compute_rsi(self._hist([100] * 10)))

    def test_none_returns_nan(self):
        assert math.isnan(compute_rsi(None))

    def test_result_in_range(self):
        rng = np.random.default_rng(42)
        prices = 100 * (1 + rng.normal(0, 0.01, 60)).cumprod()
        rsi = compute_rsi(pd.DataFrame({"Close": prices}))
        assert 0 <= rsi <= 100


class TestBsPutTheta:
    def test_negative_for_long_put(self):
        # Long put loses time value daily → theta < 0
        theta = bs_put_theta(100, 100, 0.25, 0.05, 0.20)
        assert theta < 0

    def test_seller_earns_positive(self):
        # Put seller earns -theta per day
        theta = bs_put_theta(100, 100, 0.25, 0.05, 0.20)
        assert -theta > 0

    def test_atm_more_theta_than_far_otm(self):
        atm = abs(bs_put_theta(100, 100, 0.25, 0.05, 0.20))
        otm = abs(bs_put_theta(100,  60, 0.25, 0.05, 0.20))
        assert atm > otm
