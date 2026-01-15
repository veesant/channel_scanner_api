#!/usr/bin/env python3
"""
Entry Signal Scanner (DAILY) — Range-HL Touch + Bounce Entry

(Existing behavior unchanged.)

ADDED (non-breaking):
- recent_hl_uptrend: checks last N pivot LOWS (HLs) are rising and slope > threshold
- Outputs extra JSON fields only (no removals, no logic changes to existing flags)

Install:
  pip install yfinance pandas numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datetime import datetime, timezone


try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("Missing dependency: yfinance. Install with: pip install yfinance") from e


# -----------------------------
# IO helpers
# -----------------------------

def read_tickers(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tickers file not found: {path}")
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s)
    return tickers


def json_sanitize(obj):
    """Strict JSON compliance (no NaN/inf; convert numpy/pandas scalars)."""
    if obj is None:
        return None

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]

    return obj


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            v = float(x)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def extract_ticker_df(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Robust extraction for yfinance multi-ticker downloads."""
    if data is None or data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        cols = [(f, ticker) for f in fields if (f, ticker) in data.columns]
        if not cols:
            return None
        df = data.loc[:, cols].copy()
        df.columns = [c[0] for c in df.columns]
        return df

    expected = {"Open", "High", "Low", "Close"}
    if not expected.issubset(set(data.columns)):
        return None
    return data.copy()


# -----------------------------
# Pivot detection
# -----------------------------

def find_pivot_highs(high: pd.Series, n: int) -> List[pd.Timestamp]:
    """Pivot high at i if high[i] is strictly-unique max in [i-n, i+n]."""
    idxs: List[pd.Timestamp] = []
    if len(high) < 2 * n + 1:
        return idxs

    vals = high.values
    for i in range(n, len(high) - n):
        window = vals[i - n: i + n + 1]
        m = np.nanmax(window)
        if np.isnan(m):
            continue
        if vals[i] == m and np.sum(window == m) == 1:
            idxs.append(high.index[i])
    return idxs


def find_pivot_lows(low: pd.Series, n: int) -> List[pd.Timestamp]:
    """Pivot low at i if low[i] is strictly-unique min in [i-n, i+n]."""
    idxs: List[pd.Timestamp] = []
    if len(low) < 2 * n + 1:
        return idxs

    vals = low.values
    for i in range(n, len(low) - n):
        window = vals[i - n: i + n + 1]
        m = np.nanmin(window)
        if np.isnan(m):
            continue
        if vals[i] == m and np.sum(window == m) == 1:
            idxs.append(low.index[i])
    return idxs


@dataclass
class PivotPoint:
    ts: pd.Timestamp
    kind: str  # "H" or "L"
    price: float


def build_pivot_sequence(df: pd.DataFrame, pivot_n: int) -> Tuple[List[PivotPoint], List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Returns:
      - compressed, time-ordered pivot sequence (noise-reduced)
      - raw pivot highs timestamps
      - raw pivot lows timestamps
    """
    highs = find_pivot_highs(df["High"], pivot_n)
    lows = find_pivot_lows(df["Low"], pivot_n)

    pts: List[PivotPoint] = []
    for ts in highs:
        v = safe_float(df.at[ts, "High"])
        if v is not None:
            pts.append(PivotPoint(ts=ts, kind="H", price=v))
    for ts in lows:
        v = safe_float(df.at[ts, "Low"])
        if v is not None:
            pts.append(PivotPoint(ts=ts, kind="L", price=v))

    pts.sort(key=lambda p: p.ts)

    # Compress consecutive same-kind pivots: keep the more extreme
    compressed: List[PivotPoint] = []
    for p in pts:
        if not compressed:
            compressed.append(p)
            continue

        last = compressed[-1]
        if p.kind != last.kind:
            compressed.append(p)
            continue

        if p.kind == "H":
            if p.price > last.price:
                compressed[-1] = p
        else:
            if p.price < last.price:
                compressed[-1] = p

    return compressed, highs, lows


# -----------------------------
# Trend quality helpers
# -----------------------------

def linear_slope(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return 0.0
    a, _b = np.polyfit(x[mask], y[mask], 1)
    return float(a)


def max_drawdown_pct(close: np.ndarray) -> float:
    if len(close) < 2:
        return 0.0
    c = close.astype(float)
    mask = ~np.isnan(c)
    if mask.sum() < 2:
        return 0.0
    c = c[mask]

    peak = -np.inf
    mdd = 0.0
    for v in c:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak * 100.0
            if dd > mdd:
                mdd = dd
    return float(mdd)


# -----------------------------
# Clean HH/HL uptrend definition (structure gate)
# -----------------------------

def is_clean_hh_hl_uptrend(
    df: pd.DataFrame,
    pivots: List[PivotPoint],
    structure_bars: int,
    min_pivots_per_side: int,
    sma_fast: int,
    sma_slow: int,
    require_sma_trend: bool,
    require_positive_slope: bool,
    max_dd_cap_pct: Optional[float],
) -> Tuple[bool, Dict[str, Optional[float]]]:
    if df is None or df.empty:
        return False, {}

    window = df.tail(structure_bars).copy()
    if len(window) < max(10, structure_bars // 2):
        return False, {}

    # SMA
    window["SMA_FAST"] = window["Close"].rolling(sma_fast).mean()
    window["SMA_SLOW"] = window["Close"].rolling(sma_slow).mean()
    last_close = safe_float(window["Close"].iloc[-1])
    last_sma_fast = safe_float(window["SMA_FAST"].iloc[-1])
    last_sma_slow = safe_float(window["SMA_SLOW"].iloc[-1])

    # pivots within structure window
    start_ts = window.index[0]
    end_ts = window.index[-1]
    pts = [p for p in pivots if start_ts <= p.ts <= end_ts]

    highs = [p for p in pts if p.kind == "H"]
    lows = [p for p in pts if p.kind == "L"]

    if len(highs) < min_pivots_per_side or len(lows) < min_pivots_per_side:
        return False, {
            "structure_high_pivots": float(len(highs)),
            "structure_low_pivots": float(len(lows)),
        }

    # alternation check
    for i in range(1, len(pts)):
        if pts[i].kind == pts[i - 1].kind:
            return False, {"pivot_alternation_ok": 0.0}

    # HH chain / HL chain
    high_prices = [p.price for p in highs]
    low_prices = [p.price for p in lows]

    hh_chain_ok = all(high_prices[i] > high_prices[i - 1] for i in range(1, len(high_prices)))
    hl_chain_ok = all(low_prices[i] > low_prices[i - 1] for i in range(1, len(low_prices)))

    if not (hh_chain_ok and hl_chain_ok):
        return False, {
            "hh_chain_ok": 1.0 if hh_chain_ok else 0.0,
            "hl_chain_ok": 1.0 if hl_chain_ok else 0.0,
        }

    # SMA gate
    sma_ok = True
    if require_sma_trend:
        sma_ok = (
            last_close is not None
            and last_sma_fast is not None
            and last_sma_slow is not None
            and last_sma_fast > last_sma_slow
            and last_close > last_sma_slow
        )
        if not sma_ok:
            return False, {
                "sma_ok": 0.0,
                "last_close": last_close,
                "last_sma_fast": last_sma_fast,
                "last_sma_slow": last_sma_slow,
            }

    # slope gate
    slope = linear_slope(window["Close"].values)
    if require_positive_slope and not (slope > 0):
        return False, {"close_slope": slope, "slope_ok": 0.0}

    # drawdown cap
    dd = max_drawdown_pct(window["Close"].values)
    if max_dd_cap_pct is not None and dd > max_dd_cap_pct:
        return False, {"max_drawdown_pct": dd, "max_dd_ok": 0.0}

    return True, {
        "structure_points": float(len(pts)),
        "structure_high_pivots": float(len(highs)),
        "structure_low_pivots": float(len(lows)),
        "hh_chain_ok": 1.0,
        "hl_chain_ok": 1.0,
        "close_slope": slope,
        "max_drawdown_pct": dd,
        "sma_ok": 1.0 if sma_ok else 0.0,
        "last_close": last_close,
        "last_sma_fast": last_sma_fast,
        "last_sma_slow": last_sma_slow,
    }


# -----------------------------
# Entry helpers (range-HL touch + bounce)
# -----------------------------

def pct_above(level: Optional[float], price: Optional[float]) -> Optional[float]:
    """Percent above level: (price - level)/level*100."""
    if level is None or price is None or level <= 0:
        return None
    return (price - level) / level * 100.0


# -----------------------------
# ADDED: Recent HL Uptrend (last few HLs rising + upward slope)
# -----------------------------

def compute_recent_hl_uptrend(
    df: pd.DataFrame,
    pivot_lows: List[pd.Timestamp],
    lookback_bars: int,
    hl_count: int,
    slope_min_pct_per_bar: float,
    require_strict_higher_lows: bool,
) -> Dict[str, Optional[object]]:
    """
    Uses last `hl_count` pivot lows within the last `lookback_bars` bars.
    Returns fields to add to JSON without affecting existing logic.
    """
    out: Dict[str, Optional[object]] = {
        "recent_hl_uptrend": False,
        "recent_hl_count_required": hl_count,
        "recent_hl_lookback_bars": lookback_bars,
        "recent_hl_pivots_found": 0,
        "recent_hl_prices": None,
        "recent_hl_dates": None,
        "recent_hl_chain_ok": None,
        "recent_hl_slope": None,                 # absolute price slope per bar
        "recent_hl_slope_pct_per_bar": None,     # slope normalized by last HL price
        "recent_hl_slope_min_pct_per_bar": slope_min_pct_per_bar,
    }

    if df is None or df.empty or lookback_bars <= 0 or hl_count <= 1:
        return out

    # Restrict pivots to lookback window
    w = df.tail(lookback_bars)
    start_ts = w.index[0]
    end_ts = w.index[-1]
    lows_in_window = [ts for ts in pivot_lows if start_ts <= ts <= end_ts]

    if len(lows_in_window) < hl_count:
        out["recent_hl_pivots_found"] = int(len(lows_in_window))
        return out

    last_lows = lows_in_window[-hl_count:]
    prices: List[float] = []
    dates: List[str] = []
    x: List[int] = []

    # Map timestamps to bar positions in the lookback window for slope calc
    index_pos = {ts: i for i, ts in enumerate(w.index)}

    for ts in last_lows:
        v = safe_float(df.at[ts, "Low"])
        if v is None:
            return out
        prices.append(v)
        dates.append(str(ts.date()))
        x.append(index_pos.get(ts, None))

    # If any pivot timestamp isn't in the window index (shouldn't happen), bail safely
    if any(v is None for v in x):
        out["recent_hl_pivots_found"] = int(len(last_lows))
        out["recent_hl_prices"] = prices
        out["recent_hl_dates"] = dates
        return out

    out["recent_hl_pivots_found"] = int(len(last_lows))
    out["recent_hl_prices"] = prices
    out["recent_hl_dates"] = dates

    # Chain check: higher lows
    if require_strict_higher_lows:
        chain_ok = all(prices[i] > prices[i - 1] for i in range(1, len(prices)))
    else:
        chain_ok = all(prices[i] >= prices[i - 1] for i in range(1, len(prices)))
    out["recent_hl_chain_ok"] = bool(chain_ok)

    # Slope of HLs: linear regression on (bar_index, hl_price)
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(prices, dtype=float)
    if len(x_arr) >= 2:
        a, _b = np.polyfit(x_arr, y_arr, 1)  # price units per bar
        slope_abs = float(a)
        out["recent_hl_slope"] = slope_abs

        last_hl = prices[-1]
        if last_hl > 0:
            out["recent_hl_slope_pct_per_bar"] = float((slope_abs / last_hl) * 100.0)

    slope_pct_per_bar = out["recent_hl_slope_pct_per_bar"]
    slope_ok = (slope_pct_per_bar is not None) and (slope_pct_per_bar >= slope_min_pct_per_bar)

    out["recent_hl_uptrend"] = bool(chain_ok and slope_ok)
    return out


# -----------------------------
# ADDED: Ready-to-Act flag
#   - Uptrend recently (uses existing clean trend gate: has_hh_hl)
#   - Pulled back from most recent pivot HH
#   - Current close is above the *second most recent* pivot HH
# -----------------------------

def compute_ready_to_act(
    df: pd.DataFrame,
    pivot_highs: List[pd.Timestamp],
    pivot_lows: List[pd.Timestamp],
    require_clean_trend: bool,
    clean_ok: bool,
    pullback_min_pct: float,
    pullback_lookback_bars: int,
) -> Dict[str, Optional[object]]:
    """
    Non-breaking add-on flag.

    READY-TO-ACT is True when:
      1) (optional) clean uptrend gate is satisfied (has_hh_hl)
      2) after the most recent pivot HH, price retraced (structure pullback)
      3) current close is above the second most recent pivot HH

    Pullback is detected only when, after the most recent pivot HH, price dips below the
    second most recent pivot HH (using intraday Low) and then reclaims it (enforced by
    requiring current close > second-last HH).
    """

    out: Dict[str, Optional[object]] = {
        "ready_to_act": False,

        # Debug fields (helpful for UI)
        "rta_second_last_hh": None,
        "rta_last_hh": None,
        "rta_pullback_ok": None,
        "rta_pullback_pct_from_last_hh": None,
        "rta_pullback_min_pct": pullback_min_pct,
        "rta_pullback_lookback_bars": pullback_lookback_bars,
    }

    if df is None or df.empty:
        return out

    last_close = safe_float(df["Close"].iloc[-1])
    if last_close is None:
        return out

    # Need at least 2 pivot highs for "close above second most recent HH"
    if len(pivot_highs) < 2:
        return out

    second_last_hh_ts = pivot_highs[-2]
    last_hh_ts = pivot_highs[-1]

    second_last_hh = safe_float(df.at[second_last_hh_ts, "High"])
    last_hh = safe_float(df.at[last_hh_ts, "High"])

    out["rta_second_last_hh"] = second_last_hh
    out["rta_last_hh"] = last_hh

    if second_last_hh is None or last_hh is None or last_hh <= 0:
        return out

    # 1) Trend gate (reuse existing clean uptrend)
    if require_clean_trend and not clean_ok:
        return out

    # 2) Pullback gate (DIP BELOW 2ND-LAST HH + RECLAIM)
    # Require price to actually DIP below the second-last HH after the most recent HH.
    # Reclaim is enforced later by requiring last_close > second_last_hh.
    pullback_ok = False

    if pullback_lookback_bars > 0:
        w = df.tail(pullback_lookback_bars)
        if not w.empty:
            # Prefer measuring from the HH bar onward if it's within the lookback window.
            # If HH is older than the window, we still evaluate the window (conservative).
            if last_hh_ts in w.index:
                w2 = w.loc[last_hh_ts:]
            else:
                w2 = w

            if not w2.empty:
                # Keep the existing debug metric: pullback % from last HH using min CLOSE.
                min_close = safe_float(w2["Close"].min())
                if min_close is not None and last_hh > 0:
                    pullback_pct = (last_hh - min_close) / last_hh * 100.0
                    out["rta_pullback_pct_from_last_hh"] = pullback_pct

                # Actual pullback condition: price dipped below the second-last HH
                # (use intraday Low to capture real dips, not just closes).
                min_low = safe_float(w2["Low"].min())
                if min_low is not None:
                    pullback_ok = bool(min_low < second_last_hh)

    out["rta_pullback_ok"] = bool(pullback_ok)

    # 3) Close above second-last HH
    close_above_second_last_hh = bool(last_close > second_last_hh)

    out["ready_to_act"] = bool(pullback_ok and close_above_second_last_hh)
    return out


# -----------------------------
# Scan per ticker
# -----------------------------

def scan_one(ticker: str, data: pd.DataFrame, args: argparse.Namespace) -> Optional[Dict]:
    df = extract_ticker_df(data, ticker)
    if df is None or df.empty:
        return None

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        return None

    df = df.tail(args.max_trading_days).copy()
    if len(df) < max(10, args.structure_bars // 2):
        return None

    # Range levels over structure window (TradingView-style)
    w = df.tail(args.structure_bars)
    range_hh = safe_float(w["High"].max())
    range_hl = safe_float(w["Low"].min())

    # Pivot levels (for debug/visibility)
    pivot_seq, raw_highs, raw_lows = build_pivot_sequence(df, args.pivot)
    pivot_last_hh = safe_float(df.at[raw_highs[-1], "High"]) if len(raw_highs) >= 1 else None
    pivot_last_hl = safe_float(df.at[raw_lows[-1], "Low"]) if len(raw_lows) >= 1 else None

    # Trend gate (always pivot-based structure)
    clean_ok, metrics = is_clean_hh_hl_uptrend(
        df=df,
        pivots=pivot_seq,
        structure_bars=args.structure_bars,
        min_pivots_per_side=args.min_pivots_per_side,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        require_sma_trend=(not args.disable_sma_gate),
        require_positive_slope=(not args.disable_slope_gate),
        max_dd_cap_pct=None if args.max_drawdown_pct <= 0 else args.max_drawdown_pct,
    )

    # Latest prices
    last_open = safe_float(df["Open"].iloc[-1])
    last_high = safe_float(df["High"].iloc[-1])
    last_low = safe_float(df["Low"].iloc[-1])
    last_close = safe_float(df["Close"].iloc[-1])
    prev_close = safe_float(df["Close"].iloc[-2]) if len(df) >= 2 else None

    # SMA info (output only)
    df["SMA_FAST"] = df["Close"].rolling(args.sma_fast).mean()
    df["SMA_SLOW"] = df["Close"].rolling(args.sma_slow).mean()
    sma_fast_last = safe_float(df["SMA_FAST"].iloc[-1])
    sma_slow_last = safe_float(df["SMA_SLOW"].iloc[-1])
    sma_fast_gt_slow = None
    if sma_fast_last is not None and sma_slow_last is not None:
        sma_fast_gt_slow = bool(sma_fast_last > sma_slow_last)

    # Entry levels
    if args.use_range_levels_for_entry:
        entry_level_source = "range"
        entry_hl = range_hl
        entry_hh = range_hh
    else:
        # If you don't use range levels, we still compute entry using pivot HL by default (legacy behavior)
        entry_level_source = "pivot"
        entry_hl = pivot_last_hl
        entry_hh = pivot_last_hh

    # --- Distances relative to ENTRY HL ---
    # Touch logic uses LOW; bounce logic uses CLOSE
    hl_touch_pct = pct_above(entry_hl, last_low)     # how far LOW is above HL
    close_vs_hl_pct = pct_above(entry_hl, last_close)  # how far CLOSE is above HL

    # at_hl_zone: "close to HL"
    at_hl_zone = False
    if clean_ok and hl_touch_pct is not None:
        at_hl_zone = (hl_touch_pct >= 0.0) and (hl_touch_pct <= args.hl_touch_pct)

    # bounce_ok: close is at least bounce_min_pct above HL
    bounce_ok = False
    if clean_ok and close_vs_hl_pct is not None:
        bounce_ok = close_vs_hl_pct >= args.bounce_min_pct

    # momentum_ok: simple up-day confirmation
    momentum_ok = False
    if clean_ok and last_close is not None:
        momentum_ok = (
            (last_open is not None and last_close > last_open) or
            (prev_close is not None and last_close > prev_close)
        )

    # entry_signal: trend + touched HL + bounced + momentum
    entry_signal = bool(clean_ok and at_hl_zone and bounce_ok and momentum_ok)

    # Keep breakout distance as a reference only (NOT used for entry)
    breakout_distance_pct = None
    if entry_hh is not None and last_close is not None and entry_hh > 0:
        breakout_distance_pct = (entry_hh - last_close) / entry_hh * 100.0

    # ADDED: Recent HL uptrend fields (independent; does not change existing flags)
    recent_hl_fields = compute_recent_hl_uptrend(
        df=df,
        pivot_lows=raw_lows,
        lookback_bars=args.recent_hl_lookback_bars,
        hl_count=args.recent_hl_count,
        slope_min_pct_per_bar=args.recent_hl_slope_min_pct_per_bar,
        require_strict_higher_lows=(not args.recent_hl_allow_equal_lows),
    )

    # ADDED: Ready-to-act fields (independent; does not change existing flags)
    ready_to_act_fields = compute_ready_to_act(
        df=df,
        pivot_highs=raw_highs,
        pivot_lows=raw_lows,
        require_clean_trend=True,
        clean_ok=clean_ok,
        pullback_min_pct=args.rta_pullback_min_pct,
        pullback_lookback_bars=args.rta_pullback_lookback_bars,
    )

    last_date = str(df.index[-1].date()) if len(df.index) else None

    out = {
        "ticker": ticker,
        "last_date": last_date,

        "last_open": last_open,
        "last_high": last_high,
        "last_low": last_low,
        "last_close": last_close,

        # Trend qualification (structure)
        "has_hh_hl": bool(clean_ok),
        "structure_bars": args.structure_bars,
        "min_pivots_per_side": args.min_pivots_per_side,

        # Levels (for transparency)
        "pivot_last_hh": pivot_last_hh,
        "pivot_last_hl": pivot_last_hl,
        "range_hh": range_hh,
        "range_hl": range_hl,

        # Entry levels used
        "entry_level_source": entry_level_source,  # "range" or "pivot"
        "last_hl": entry_hl,
        "last_hh": entry_hh,

        # Core flags
        "at_hl_zone": bool(at_hl_zone),
        "bounce_ok": bool(bounce_ok),
        "momentum_ok": bool(momentum_ok),
        "entry_signal": bool(entry_signal),

        # Debug distances
        "hl_touch_pct": hl_touch_pct,
        "close_vs_hl_pct": close_vs_hl_pct,
        "breakout_distance_pct": breakout_distance_pct,

        # Thresholds used
        "hl_touch_pct_threshold": args.hl_touch_pct,
        "bounce_min_pct_threshold": args.bounce_min_pct,

        # SMA info
        "sma_fast": args.sma_fast,
        "sma_slow": args.sma_slow,
        "sma_fast_last": sma_fast_last,
        "sma_slow_last": sma_slow_last,
        "sma_fast_gt_slow": sma_fast_gt_slow,

        # Trend debug metrics
        **{f"metric_{k}": v for k, v in metrics.items()},
    }

    # Append new fields (no removals)
    out.update(recent_hl_fields)
    out.update(ready_to_act_fields)
    return out


# -----------------------------
# CLI / Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entry Signal Scanner (Range-HL touch + Bounce entry)")

    p.add_argument("--tickers_file", default="tickers.txt", help="Text file with tickers, one per line")
    p.add_argument("--out", default="output/entry_signals.json", help="Output JSON file path")

    p.add_argument("--last_days", type=int, default=150, help="Calendar days to download")
    p.add_argument("--max_trading_days", type=int, default=100, help="Keep last N trading days after download")

    p.add_argument("--pivot", type=int, default=3, help="Pivot window n (uses [i-n..i+n])")
    p.add_argument("--structure_bars", type=int, default=35, help="Bars used to validate clean uptrend")
    p.add_argument("--min_pivots_per_side", type=int, default=3, help="Min pivot highs and lows in structure window")

    p.add_argument("--sma_fast", type=int, default=10, help="Fast SMA")
    p.add_argument("--sma_slow", type=int, default=20, help="Slow SMA")
    p.add_argument("--disable_sma_gate", action="store_true", help="Disable SMA gate (fast>slow and close>slow)")
    p.add_argument("--disable_slope_gate", action="store_true", help="Disable positive slope requirement")
    p.add_argument("--max_drawdown_pct", type=float, default=12.0, help="Max drawdown %% allowed (<=0 disables)")

    # Existing entry controls (unchanged)
    p.add_argument("--use_range_levels_for_entry", action="store_true",
                   help="Use range_hl/range_hh as entry levels (range_hl touch + bounce entry)")

    # NOTE: keeping YOUR current default as-is to avoid changing behavior
    p.add_argument("--hl_touch_pct", type=float, default=100,
                   help="at_hl_zone TRUE if today's LOW is within this %% above HL (0.6 means <=0.6%%)")

    p.add_argument("--bounce_min_pct", type=float, default=0.3,
                   help="bounce_ok TRUE if CLOSE is at least this %% above HL (0.3 means >=0.3%%)")

    # ADDED: recent HL uptrend filter knobs (new only; no existing behavior changes)
    p.add_argument("--recent_hl_lookback_bars", type=int, default=20,
                   help="Lookback bars to search for recent HL pivots (ascending trendline)")
    p.add_argument("--recent_hl_count", type=int, default=3,
                   help="Number of most recent HL pivots required (e.g., 3 means last 3 pivot lows)")
    p.add_argument("--recent_hl_slope_min_pct_per_bar", type=float, default=0.02,
                   help="Minimum HL slope in %% per bar (0.02 = 0.02%% per bar)")
    p.add_argument("--recent_hl_allow_equal_lows", action="store_true",
                   help="Allow equal pivot lows (>=) instead of strictly higher lows (>)")

    # ADDED: ready-to-act knobs (new only; does not affect existing behavior)
    p.add_argument("--rta_pullback_min_pct", type=float, default=2.0,
                   help="Min pullback %% from most recent pivot HH to qualify as a pullback (default 2.0)")
    p.add_argument("--rta_pullback_lookback_bars", type=int, default=20,
                   help="Lookback bars used to measure pullback from last HH (default 20)")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers = read_tickers(args.tickers_file)
    if not tickers:
        print("No tickers found.", file=sys.stderr)
        return 2

    # ============================
    # Yahoo Finance download (SAFE)
    # ============================

    start_ts = pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=int(args.last_days))

    import time
    import random

    def _is_rate_limited(err):
        msg = str(err).lower()
        return (
            "too many requests" in msg
            or "rate limited" in msg
            or "429" in msg
        )

    def _download_batch(batch):
        return yf.download(
            tickers=batch,
            start=start_ts,
            interval="1d",
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=False,   # IMPORTANT: disable threading
        )

    def _download_with_backoff(batch, max_retries=6, base_sleep=2.0):
        last_err = None
        for attempt in range(max_retries):
            try:
                return _download_batch(batch)
            except Exception as e:
                last_err = e
                if not _is_rate_limited(e) or attempt == max_retries - 1:
                    raise
                sleep_s = min(
                    120.0,
                    base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0)
                )
                time.sleep(sleep_s)
        raise last_err

    BATCH_SIZE = 40   # Safe for ~3,100 NASDAQ tickers
    frames = []

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = [t.strip().upper() for t in tickers[i:i + BATCH_SIZE] if t.strip()]
        if not batch:
            continue

        try:
            df_batch = _download_with_backoff(batch)
            if df_batch is not None and not df_batch.empty:
                frames.append(df_batch)
        except Exception as e:
            # If Yahoo throttles a batch, fall back to single-ticker downloads
            if _is_rate_limited(e):
                for t in batch:
                    try:
                        df_one = _download_with_backoff([t])
                        if df_one is not None and not df_one.empty:
                            frames.append(df_one)
                        time.sleep(0.3 + random.uniform(0.0, 0.3))
                    except Exception:
                        pass
            else:
                raise

        # Gentle pacing between batches
        time.sleep(0.8 + random.uniform(0.0, 0.5))

    data = pd.concat(frames, axis=1) if frames else pd.DataFrame()

    # ============================
    # End Yahoo Finance download
    # ============================


    rows: List[Dict] = []
    for t in tickers:
        try:
            rec = scan_one(t, data, args)
            if rec is not None:
                rows.append(rec)
        except Exception as e:
            rows.append({"ticker": t, "has_hh_hl": False, "entry_signal": False, "error": str(e)})

    df = pd.DataFrame(rows)

    if not df.empty:
        for c in ["entry_signal", "has_hh_hl", "at_hl_zone", "bounce_ok", "momentum_ok", "recent_hl_uptrend", "ready_to_act"]:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)

        # Keep your existing sort columns (unchanged)
        sort_cols = ["entry_signal", "at_hl_zone", "recent_hl_uptrend", "has_hh_hl", "hl_touch_pct", "close_vs_hl_pct", "last_date"]
        existing = [c for c in sort_cols if c in df.columns]
        if existing:
            # prefer True first; for pct columns sort ascending; date descending
            ascending = []
            for c in existing:
                if c in ("hl_touch_pct", "close_vs_hl_pct"):
                    ascending.append(True)
                elif c == "last_date":
                    ascending.append(False)
                else:
                    ascending.append(False)
            df = df.sort_values(by=existing, ascending=ascending, kind="mergesort")

    payload = {
        "meta": {
            "tickers_file": args.tickers_file,
            "last_days": args.last_days,
            "max_trading_days": args.max_trading_days,
            "pivot": args.pivot,
            "structure_bars": args.structure_bars,
            "min_pivots_per_side": args.min_pivots_per_side,
            "sma_fast": args.sma_fast,
            "sma_slow": args.sma_slow,
            "disable_sma_gate": bool(args.disable_sma_gate),
            "disable_slope_gate": bool(args.disable_slope_gate),
            "max_drawdown_pct": args.max_drawdown_pct,
            "use_range_levels_for_entry": bool(args.use_range_levels_for_entry),
            "hl_touch_pct": args.hl_touch_pct,
            "bounce_min_pct": args.bounce_min_pct,

            # ADDED meta
            "recent_hl_lookback_bars": args.recent_hl_lookback_bars,
            "recent_hl_count": args.recent_hl_count,
            "recent_hl_slope_min_pct_per_bar": args.recent_hl_slope_min_pct_per_bar,
            "recent_hl_allow_equal_lows": bool(args.recent_hl_allow_equal_lows),

            # ADDED meta (ready-to-act)
            "rta_pullback_min_pct": args.rta_pullback_min_pct,
            "rta_pullback_lookback_bars": args.rta_pullback_lookback_bars,
            "lastUpdatedTs": datetime.now(timezone.utc).isoformat(),
        },
        "count": int(len(df)),
        "count_clean_trend": int(df["has_hh_hl"].sum()) if (not df.empty and "has_hh_hl" in df.columns) else 0,
        "count_entry_signal": int(df["entry_signal"].sum()) if (not df.empty and "entry_signal" in df.columns) else 0,
        "count_recent_hl_uptrend": int(df["recent_hl_uptrend"].sum()) if (not df.empty and "recent_hl_uptrend" in df.columns) else 0,
        "count_ready_to_act": int(df["ready_to_act"].sum()) if (not df.empty and "ready_to_act" in df.columns) else 0,
        "data": df.to_dict(orient="records") if not df.empty else [],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload_clean = json_sanitize(payload)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload_clean, f, indent=2, allow_nan=False)

    print(f"\nSaved: {args.out}")

    if df.empty:
        print("No results.")
        return 0

    preview_cols = [
        "ticker", "entry_signal", "ready_to_act", "has_hh_hl", "at_hl_zone", "bounce_ok", "momentum_ok",
        "recent_hl_uptrend", "recent_hl_pivots_found", "recent_hl_slope_pct_per_bar",
        "entry_level_source",
        "hl_touch_pct", "close_vs_hl_pct",
        "range_hl", "last_hl",
        "last_low", "last_close",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("\nPreview (top 25):")
    print(df[preview_cols].head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
