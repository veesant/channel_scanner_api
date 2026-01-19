#!/usr/bin/env python3
"""
Actionable-after-correction scanner (structure-first, TradingView-like)

Key updates (Most recent levels):
- recent_swing_high is now the MOST RECENT pivot high (not the max-high in the window)
- resistance_level = recent_swing_high
- support_level (anchor) = MOST RECENT pivot low BEFORE that swing high (within support_lookback)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) and len(c) else str(c) for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    if any(c not in df.columns for c in REQUIRED_COLS):
        return pd.DataFrame()

    df = df[REQUIRED_COLS].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index().dropna()


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = normalize_ohlc(df)
    if df.empty:
        return df
    return (
        df.resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )

#price quality check

def price_quality_gate(
    df: pd.DataFrame,
    n_bars: int = 50,
    # wick rules
    long_wick_ratio: float = 0.45,
    max_body_ratio: float = 0.35,
    max_long_wicks: int = 6,
    # gap rules
    gap_thresh_pct: float = 2.5,
    max_gap_count: int = 2,
    max_gap_pct: float = 6.0,
):
    """
    Returns: (ok: bool, diagnostics: dict)
    """
    d = df.tail(n_bars).copy()
    if len(d) < 3:
        return True, {"note": "not_enough_bars_for_quality_gate"}

    o = d["Open"].astype(float).values
    h = d["High"].astype(float).values
    l = d["Low"].astype(float).values
    c = d["Close"].astype(float).values

    rng = h - l
    rng_safe = np.where(rng <= 0, np.nan, rng)

    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    max_wick = np.maximum(upper, lower)
    max_wick_ratio_arr = max_wick / rng_safe
    body_ratio_arr = body / rng_safe

    long_wick_bar = (max_wick_ratio_arr >= long_wick_ratio) & (body_ratio_arr <= max_body_ratio)
    long_wick_count = int(np.nansum(long_wick_bar))

    # gaps vs previous close (align within the tail window)
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    gap_pct = np.abs(o - prev_c) / prev_c * 100.0
    gap_count = int(np.nansum(gap_pct >= gap_thresh_pct))
    max_gap = float(np.nanmax(gap_pct)) if np.isfinite(np.nanmax(gap_pct)) else None

    ok = True
    if long_wick_count > max_long_wicks:
        ok = False
    if gap_count > max_gap_count:
        ok = False
    if max_gap is not None and max_gap >= max_gap_pct:
        ok = False

    avg_volume_50 = None
    if "Volume" in d.columns:
        avg_volume_50 = int(d["Volume"].mean())


    diag = {
        "quality_n_bars": int(len(d)),
        "long_wick_count": long_wick_count,
        "max_long_wicks_allowed": int(max_long_wicks),
        "gap_thresh_pct": float(gap_thresh_pct),
        "gap_count": gap_count,
        "max_gap_count_allowed": int(max_gap_count),
        "max_gap_pct": max_gap,
        "max_gap_pct_allowed": float(max_gap_pct),
        "avg_volume_50": avg_volume_50,
        "price_quality_ok": bool(ok),
    }
    return ok, diag

def download_bars(ticker: str, tf: str, lookback_days: int, threads: bool = True) -> pd.DataFrame:
    import time
    import random

    def is_rate_limited(err) -> bool:
        msg = str(err).lower()
        return (
            "too many requests" in msg
            or "rate limited" in msg
            or "429" in msg
        )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    t = ticker.upper().strip()
    tf_l = tf.lower().strip()

    max_retries = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            if tf_l in ("4h", "2h"):
                raw = yf.download(
                    t,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="60m",
                    auto_adjust=True,
                    progress=False,
                    threads=False,   # IMPORTANT
                )
                raw = normalize_ohlc(raw)
                if raw.empty:
                    return pd.DataFrame()
                rule = "4h" if tf_l == "4h" else "2h"
                return resample_ohlc(raw, rule)

            interval = "1d" if tf_l in ("1d", "d", "day") else tf_l
            raw = yf.download(
                t,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,   # IMPORTANT
            )
            return normalize_ohlc(raw)

        except Exception as e:
            last_err = e
            if not is_rate_limited(e) or attempt == max_retries - 1:
                raise

            # Exponential backoff + jitter
            sleep_s = min(
                120.0,
                base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0)
            )
            time.sleep(sleep_s)

    raise last_err



def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def read_tickers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out


def _pivot_points(values: np.ndarray, left: int, right: int, mode: str) -> List[Tuple[int, float]]:
    """
    Simple pivot detector:
      - pivot high: values[i] is the max in [i-left, i+right]
      - pivot low : values[i] is the min in [i-left, i+right]
    Returns list of (index, value) in ascending index order.
    """
    n = len(values)
    if n == 0:
        return []
    left = int(max(1, left))
    right = int(max(1, right))

    pivots: List[Tuple[int, float]] = []
    for i in range(left, n - right):
        window = values[i - left : i + right + 1]
        v = values[i]
        if not np.isfinite(v):
            continue

        if mode == "high":
            m = np.nanmax(window)
            if np.isfinite(m) and v == m and np.sum(window == m) == 1:
                pivots.append((i, float(v)))
        elif mode == "low":
            m = np.nanmin(window)
            if np.isfinite(m) and v == m and np.sum(window == m) == 1:
                pivots.append((i, float(v)))
        else:
            raise ValueError("mode must be 'high' or 'low'")
    return pivots


def _recent_pivots_in_tail(df: pd.DataFrame, lookback: int, pivot_strength: int) -> Dict[str, Any]:
    """
    Return pivot highs/lows found inside the last `lookback` bars, in ABSOLUTE df positions.
    """
    lookback = int(max(10, lookback))
    pivot_strength = int(max(1, pivot_strength))

    tail = df.tail(lookback)
    if tail.empty:
        return {"ph": [], "pl": [], "lookback_used": 0}

    highs = tail["High"].values.astype(float)
    lows = tail["Low"].values.astype(float)

    ph_local = _pivot_points(highs, left=pivot_strength, right=pivot_strength, mode="high")
    pl_local = _pivot_points(lows, left=pivot_strength, right=pivot_strength, mode="low")

    offset = len(df) - len(tail)  # local index -> absolute position index
    ph_abs = [(offset + i, v) for i, v in ph_local]
    pl_abs = [(offset + i, v) for i, v in pl_local]

    return {"ph": ph_abs, "pl": pl_abs, "lookback_used": len(tail)}


def _most_recent_pivot_high(df: pd.DataFrame, lookback: int, pivot_strength: int) -> Tuple[Optional[int], Optional[float]]:
    piv = _recent_pivots_in_tail(df, lookback=lookback, pivot_strength=pivot_strength)
    ph = piv["ph"]
    if not ph:
        return None, None
    pos, val = ph[-1]  # most recent pivot high
    return int(pos), float(val)


def _most_recent_pivot_low_before(
    df: pd.DataFrame, end_pos_exclusive: int, lookback: int, pivot_strength: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the most recent pivot LOW strictly before end_pos_exclusive, scanning within a lookback window.
    """
    lookback = int(max(10, lookback))
    pivot_strength = int(max(1, pivot_strength))

    start = max(0, end_pos_exclusive - lookback)
    seg = df.iloc[start:end_pos_exclusive]
    if len(seg) < (pivot_strength * 2 + 3):
        return None, None

    lows = seg["Low"].values.astype(float)
    pl_local = _pivot_points(lows, left=pivot_strength, right=pivot_strength, mode="low")
    if not pl_local:
        return None, None

    local_i, val = pl_local[-1]
    return int(start + local_i), float(val)


def compute_uptrend_hh_hl(
    df: pd.DataFrame,
    window_min: int = 30,
    window_max: int = 50,
    pivot_strength: int = 2,
    min_hh: int = 2,
    min_hl: int = 2,
) -> Dict[str, Any]:
    window_min = int(max(10, window_min))
    window_max = int(max(window_min, window_max))
    pivot_strength = int(max(1, pivot_strength))

    tail = df.tail(window_max).copy()
    if len(tail) < window_min:
        return {
            "uptrend_window_used": int(len(tail)),
            "uptrend_hh_count": 0,
            "uptrend_hl_count": 0,
            "uptrend_pivot_highs": 0,
            "uptrend_pivot_lows": 0,
            "uptrend_ok": False,
        }

    highs = tail["High"].values.astype(float)
    lows = tail["Low"].values.astype(float)
    closes = tail["Close"].values.astype(float)

    ph = _pivot_points(highs, left=pivot_strength, right=pivot_strength, mode="high")
    pl = _pivot_points(lows, left=pivot_strength, right=pivot_strength, mode="low")

    hh_count = 0
    prev_h = None
    for _, v in ph:
        if prev_h is None:
            prev_h = v
            continue
        if v > prev_h:
            hh_count += 1
        prev_h = v

    hl_count = 0
    prev_l = None
    for _, v in pl:
        if prev_l is None:
            prev_l = v
            continue
        if v > prev_l:
            hl_count += 1
        prev_l = v

    overall_up = bool(np.isfinite(closes[0]) and np.isfinite(closes[-1]) and closes[-1] > closes[0])
    uptrend_ok = bool(overall_up and hh_count >= int(min_hh) and hl_count >= int(min_hl))

    return {
        "uptrend_window_used": int(len(tail)),
        "uptrend_hh_count": int(hh_count),
        "uptrend_hl_count": int(hl_count),
        "uptrend_pivot_highs": int(len(ph)),
        "uptrend_pivot_lows": int(len(pl)),
        "uptrend_overall_up": bool(overall_up),
        "uptrend_ok": bool(uptrend_ok),
    }


def compute_actionable(
    df: pd.DataFrame,
    sma_fast: int,
    sma_slow: int,
    min_pullback_pct: float,
    swing_window: int,
    support_lookback: int,
    rebound_bars: int,
    support_tolerance_pct: float,
    uptrend_window_min: int,
    uptrend_window_max: int,
    pivot_strength: int,
    min_hh: int,
    min_hl: int,
) -> Dict[str, Any]:
    close = df["Close"]
    high = df["High"]

    close_now = float(close.iloc[-1])

    s_fast = sma(close, sma_fast)
    s_slow = sma(close, sma_slow)

    sma_fast_now = safe_float(s_fast.iloc[-1])
    sma_slow_now = safe_float(s_slow.iloc[-1])

    # Trend filter
    trend_ok = bool(sma_slow_now is not None and close_now > sma_slow_now)

    # --- MOST RECENT swing high (pivot-based) ---
    swing_window = int(max(20, swing_window))
    idx_high_pos, recent_high = _most_recent_pivot_high(df, lookback=swing_window, pivot_strength=pivot_strength)

    # Fallback: if no pivot found, fall back to max high in the window
    if idx_high_pos is None or recent_high is None:
        recent_slice = df.tail(swing_window)
        recent_high = float(recent_slice["High"].max())
        idx_high = recent_slice["High"].idxmax()
        idx_high_pos = int(df.index.get_indexer([idx_high])[0])
    else:
        idx_high = df.index[int(idx_high_pos)]

    pullback_pct = (recent_high - close_now) / recent_high * 100.0 if recent_high and recent_high > 0 else 0.0
    had_correction = pullback_pct >= float(min_pullback_pct)

    # Correction segment "after swing high"
    after_high = df.iloc[int(idx_high_pos) :]

    # Correction low using CLOSE (not wicks)
    correction_close_low = float(after_high["Close"].min()) if len(after_high) else float(close.tail(swing_window).min())

    # --- MOST RECENT support BEFORE swing high (pivot-based) ---
    support_lookback = int(max(20, support_lookback))
    sup_pos, sup_val = _most_recent_pivot_low_before(
        df, end_pos_exclusive=int(idx_high_pos), lookback=support_lookback, pivot_strength=pivot_strength
    )

    # If no pivot low found, fall back to lowest CLOSE before the swing high within lookback window
    if sup_pos is None or sup_val is None:
        before_high = df.iloc[max(0, int(idx_high_pos) - support_lookback) : int(idx_high_pos)]
        anchor_support_close = float(before_high["Close"].min()) if len(before_high) else float(close.tail(support_lookback).min())
        anchor_support_pos = None
    else:
        anchor_support_close = float(sup_val)
        anchor_support_pos = int(sup_pos)

    # Support held (allow tolerance)
    tol = float(max(0.0, support_tolerance_pct)) / 100.0
    support_floor = anchor_support_close
    support_floor_adj = support_floor * (1.0 - tol)
    support_held = bool(correction_close_low >= support_floor_adj)

    # Rebound confirmation
    close_above_fast = bool(sma_fast_now is not None and close_now > sma_fast_now)
    rebound_bars = int(max(3, rebound_bars))
    recent_closes = close.tail(rebound_bars).values
    net_up = bool(recent_closes[-1] > recent_closes[0])
    rebound_ok = bool(close_above_fast and net_up)

    actionable_correction = bool(trend_ok and had_correction and support_held and rebound_ok)

    # HH/HL uptrend logic (unchanged)
    uptrend_info = compute_uptrend_hh_hl(
        df=df,
        window_min=uptrend_window_min,
        window_max=uptrend_window_max,
        pivot_strength=pivot_strength,
        min_hh=min_hh,
        min_hl=min_hl,
    )
    uptrend_actionable = bool(uptrend_info["uptrend_ok"] and trend_ok)

    actionable = bool(actionable_correction or uptrend_actionable)

    dist_to_support = None
    if support_floor and support_floor > 0:
        dist_to_support = abs(close_now - support_floor) / support_floor

    # Most recent resistance = most recent swing high
    resistance_level = float(recent_high) if recent_high is not None else None
    resistance_pos = int(idx_high_pos) if idx_high_pos is not None else None

    return {
        "close": close_now,
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "sma_fast_now": safe_float(sma_fast_now),
        "sma_slow_now": safe_float(sma_slow_now),

        "trend_ok": bool(trend_ok),

        # Resistance (most recent swing high)
        "recent_swing_high": safe_float(resistance_level),
        "recent_swing_high_date": str(df.index[resistance_pos].date()) if resistance_pos is not None else None,
        "resistance_level": safe_float(resistance_level),

        "pullback_from_swing_high_pct": safe_float(pullback_pct),
        "had_correction": bool(had_correction),

        # Support (most recent pivot low before swing high)
        "anchor_support_close": safe_float(anchor_support_close),
        "anchor_support_date": str(df.index[anchor_support_pos].date()) if anchor_support_pos is not None else None,
        "support_level": safe_float(anchor_support_close),

        "support_tolerance_pct": float(support_tolerance_pct),
        "support_floor": safe_float(support_floor),
        "support_floor_adj": safe_float(support_floor_adj),
        "correction_close_low": safe_float(correction_close_low),
        "support_held": bool(support_held),

        "close_above_sma_fast": bool(close_above_fast),
        "rebound_ok": bool(rebound_ok),

        **uptrend_info,
        "uptrend_actionable": bool(uptrend_actionable),

        "actionable_after_correction": bool(actionable),

        "dist_to_support": safe_float(dist_to_support),
        "actionable_reason": (
            "correction_setup" if actionable_correction else ("hh_hl_uptrend" if uptrend_actionable else "none")
        ),
    }


def scan(
    tickers: List[str],
    tf: str,
    lookback_days: int,
    max_bars: int,
    sma_fast: int,
    sma_slow: int,
    min_pullback_pct: float,
    swing_window: int,
    support_lookback: int,
    rebound_bars: int,
    support_tolerance_pct: float,
    uptrend_window_min: int,
    uptrend_window_max: int,
    pivot_strength: int,
    min_hh: int,
    min_hl: int,
    only_actionable: bool,
    threads: bool,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue

        df = download_bars(t, tf=tf, lookback_days=lookback_days, threads=threads)
        if df.empty:
            continue

        df = df.tail(int(max_bars)).copy()

        min_len = max(
            120,
            int(sma_slow) + 10,
            int(uptrend_window_max) + int(pivot_strength) * 2 + 5,
            int(swing_window) + int(pivot_strength) * 2 + 5,
        )
        if len(df) < min_len:
            continue

        pq_ok, pq = price_quality_gate(df, n_bars=50)

        info = compute_actionable(
            df=df,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            min_pullback_pct=min_pullback_pct,
            swing_window=swing_window,
            support_lookback=support_lookback,
            rebound_bars=rebound_bars,
            support_tolerance_pct=support_tolerance_pct,
            uptrend_window_min=uptrend_window_min,
            uptrend_window_max=uptrend_window_max,
            pivot_strength=pivot_strength,
            min_hh=min_hh,
            min_hl=min_hl,
        )

        if only_actionable and not info["actionable_after_correction"]:
            continue

        score = 0.0
        score += 1.0 if info["actionable_after_correction"] else 0.0

        pb = info.get("pullback_from_swing_high_pct") or 0.0
        score += min(float(pb) / 10.0, 0.8)

        d = info.get("dist_to_support")
        if d is not None:
            score += max(0.0, 0.5 - float(d))

        if info.get("uptrend_actionable"):
            score += 0.6
            score += min(0.2, 0.05 * float(info.get("uptrend_hh_count", 0)))
            score += min(0.2, 0.05 * float(info.get("uptrend_hl_count", 0)))

        results.append({
            "ticker": t,
            "tf": tf,
            "bars_used": int(len(df)),
            **info,

            # Price quality + volume metadata (UI can filter)
            "priceQuality": pq,
            "priceQualityOk": pq["price_quality_ok"],
            "avgVolume50": pq.get("avg_volume_50"),

            "rank_score": float(score),
        })


    results.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)

    seen = set()
    out: List[Dict[str, Any]] = []
    for r in results:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actionable-after-correction scanner -> JSON")
    p.add_argument("--tickers", nargs="*", default=[], help="Tickers list")
    p.add_argument("--tickers_file", default="", help="One ticker per line")

    p.add_argument("--tf", default="1d", help="1d or 4h recommended")
    p.add_argument("--lookback_days", type=int, default=260)
    p.add_argument("--max_bars", type=int, default=300)

    p.add_argument("--sma_fast", type=int, default=20)
    p.add_argument("--sma_slow", type=int, default=50)

    p.add_argument("--min_pullback_pct", type=float, default=3.0)

    # IMPORTANT: swing_window now means "lookback to find MOST RECENT swing high pivot"
    p.add_argument("--swing_window", type=int, default=80)

    # support_lookback now means "lookback before swing high to find MOST RECENT swing low pivot"
    p.add_argument("--support_lookback", type=int, default=80)

    p.add_argument("--rebound_bars", type=int, default=4)
    p.add_argument(
        "--support_tolerance_pct",
        type=float,
        default=1.0,
        help="Allow small undercut of anchor support (in %)",
    )

    p.add_argument("--uptrend_window_min", type=int, default=30)
    p.add_argument("--uptrend_window_max", type=int, default=50)
    p.add_argument("--pivot_strength", type=int, default=2)
    p.add_argument("--min_hh", type=int, default=2)
    p.add_argument("--min_hl", type=int, default=2)

    p.add_argument("--only_actionable", action="store_true")
    p.add_argument("--out", default="output.json")
    p.add_argument("--no_threads", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    payload: Dict[str, Any] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "tf": args.tf,
            "lookback_days": args.lookback_days,
            "max_bars": args.max_bars,
            "sma_fast": args.sma_fast,
            "sma_slow": args.sma_slow,
            "min_pullback_pct": args.min_pullback_pct,
            "swing_window": args.swing_window,
            "support_lookback": args.support_lookback,
            "rebound_bars": args.rebound_bars,
            "support_tolerance_pct": args.support_tolerance_pct,
            "uptrend_window_min": args.uptrend_window_min,
            "uptrend_window_max": args.uptrend_window_max,
            "pivot_strength": args.pivot_strength,
            "min_hh": args.min_hh,
            "min_hl": args.min_hl,
            "only_actionable": bool(args.only_actionable),
        },
        "count": 0,
        "data": [],
    }

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("No tickers provided.")
        return 2

    data = scan(
        tickers=tickers,
        tf=args.tf,
        lookback_days=args.lookback_days,
        max_bars=args.max_bars,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        min_pullback_pct=args.min_pullback_pct,
        swing_window=args.swing_window,
        support_lookback=args.support_lookback,
        rebound_bars=args.rebound_bars,
        support_tolerance_pct=args.support_tolerance_pct,
        uptrend_window_min=args.uptrend_window_min,
        uptrend_window_max=args.uptrend_window_max,
        pivot_strength=args.pivot_strength,
        min_hh=args.min_hh,
        min_hl=args.min_hl,
        only_actionable=bool(args.only_actionable),
        threads=not args.no_threads,
    )

    payload["count"] = len(data)
    payload["data"] = data

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(data)} rows to {args.out}")
    if data:
        dfp = pd.DataFrame(data)[
            [
                "ticker",
                "tf",
                "actionable_after_correction",
                "actionable_reason",
                "pullback_from_swing_high_pct",
                "trend_ok",
                "support_held",
                "rebound_ok",
                "uptrend_ok",
                "uptrend_hh_count",
                "uptrend_hl_count",
                "rank_score",
            ]
        ].head(25)
        print(dfp.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
