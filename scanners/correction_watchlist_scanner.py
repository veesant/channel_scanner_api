#!/usr/bin/env python3
"""
Correction + Watchlist Scanner (1D)

Finds stocks with:
- Clean HH/HL uptrend structure (last ~30-50 bars)
- Most recent swing high = most recent PIVOT high (confirmed)
- A correction that started recently (bars since swing high within a range)
- Support = most recent PIVOT low before swing high (with tolerance)
- Two outputs:
    1) watchlist_correction: trend+uptrend+correction+support_held but rebound NOT confirmed yet
    2) actionable_after_correction: your original "rebound_ok" entry-ready condition OR uptrend-only condition

Data source: yfinance (auto_adjust=True)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Helpers
# -----------------------------

def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns if multiple tickers are fetched together
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    need = {"Open", "High", "Low", "Close", "Volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # make naive (no tz) and sorted
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index().dropna()


def download_bars(ticker: str, tf: str, lookback_days: int, threads: bool) -> pd.DataFrame:
    # only 1d supported in this script
    if tf != "1d":
        raise ValueError("This watchlist scanner supports only tf=1d")

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

    max_retries = 6
    base_sleep = 2.0
    last_err = None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker.upper().strip(),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,   # IMPORTANT: force no threading
            )
            return normalize_ohlc(raw)

        except Exception as e:
            last_err = e
            if not is_rate_limited(e) or attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            sleep_s = min(
                120.0,
                base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0)
            )
            time.sleep(sleep_s)

    raise last_err



def read_tickers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out

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

# -----------------------------
# Pivot logic
# -----------------------------

def _pivot_points(arr: np.ndarray, left: int, right: int, mode: str) -> List[Tuple[int, float]]:
    """
    Returns list of (index, value) pivot highs/lows.
    A pivot is CONFIRMED only after 'right' bars have printed.
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    pts: List[Tuple[int, float]] = []

    if n < left + right + 1:
        return pts

    for i in range(left, n - right):
        w = a[i - left : i + right + 1]
        if not np.all(np.isfinite(w)):
            continue

        if mode == "high":
            m = float(np.max(w))
            if float(a[i]) == m and np.sum(w == m) == 1:
                pts.append((i, float(a[i])))
        else:
            m = float(np.min(w))
            if float(a[i]) == m and np.sum(w == m) == 1:
                pts.append((i, float(a[i])))

    return pts


def _most_recent_pivot_high(df: pd.DataFrame, lookback: int, pivot_strength: int) -> Tuple[Optional[int], Optional[float]]:
    """
    Most recent pivot HIGH in last `lookback` bars (returns position in full df).
    """
    lookback = int(max(20, lookback))
    pivot_strength = int(max(1, pivot_strength))

    tail = df.tail(lookback)
    highs = tail["High"].values.astype(float)

    ph = _pivot_points(highs, left=pivot_strength, right=pivot_strength, mode="high")
    if not ph:
        return None, None

    local_i, v = ph[-1]
    global_pos = int(len(df) - len(tail) + local_i)
    return global_pos, float(v)


def _most_recent_pivot_low_before(
    df: pd.DataFrame, end_pos_exclusive: int, lookback: int, pivot_strength: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    Most recent pivot LOW before end_pos_exclusive, searching backwards within lookback.
    """
    lookback = int(max(20, lookback))
    pivot_strength = int(max(1, pivot_strength))

    end_pos_exclusive = int(max(0, min(end_pos_exclusive, len(df))))
    start = max(0, end_pos_exclusive - lookback)
    seg = df.iloc[start:end_pos_exclusive]
    if len(seg) < pivot_strength * 2 + 3:
        return None, None

    lows = seg["Low"].values.astype(float)
    pl = _pivot_points(lows, left=pivot_strength, right=pivot_strength, mode="low")
    if not pl:
        return None, None

    local_i, v = pl[-1]
    global_pos = int(start + local_i)
    return global_pos, float(v)


# -----------------------------
# HH/HL uptrend structure
# -----------------------------

def compute_uptrend_hh_hl(
    df: pd.DataFrame,
    window_min: int,
    window_max: int,
    pivot_strength: int,
    min_hh: int,
    min_hl: int,
) -> Dict[str, Any]:
    window_min = int(max(10, window_min))
    window_max = int(max(window_min, window_max))
    pivot_strength = int(max(1, pivot_strength))

    tail = df.tail(window_max)
    closes = tail["Close"].values.astype(float)
    highs = tail["High"].values.astype(float)
    lows = tail["Low"].values.astype(float)

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


# -----------------------------
# Main actionable / watchlist logic
# -----------------------------

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
    # new: correction age constraints
    corr_bars_min: int,
    corr_bars_max: int,
) -> Dict[str, Any]:
    close = df["Close"]
    high = df["High"]

    close_now = float(close.iloc[-1])

    s_fast = sma(close, int(sma_fast))
    s_slow = sma(close, int(sma_slow))

    sma_fast_now = safe_float(s_fast.iloc[-1])
    sma_slow_now = safe_float(s_slow.iloc[-1])

    # Trend filter: price above SMA slow
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

    # Pullback from resistance
    pullback_pct = (recent_high - close_now) / recent_high * 100.0 if recent_high and recent_high > 0 else 0.0
    had_correction = bool(pullback_pct >= float(min_pullback_pct))

    # Correction segment "after swing high"
    after_high = df.iloc[int(idx_high_pos) :]
    bars_since_swing_high = int(max(0, len(after_high) - 1))  # 0 means swing high is last bar
    correction_age_ok = bool(int(corr_bars_min) <= bars_since_swing_high <= int(corr_bars_max))

    # Correction low using CLOSE (not wicks)
    correction_close_low = float(after_high["Close"].min()) if len(after_high) else float(close.tail(swing_window).min())

    # --- MOST RECENT support BEFORE swing high (pivot-based) ---
    support_lookback = int(max(20, support_lookback))
    sup_pos, sup_val = _most_recent_pivot_low_before(
        df, end_pos_exclusive=int(idx_high_pos), lookback=support_lookback, pivot_strength=pivot_strength
    )

    # fallback: lowest close before the swing high (within lookback)
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

    # Rebound confirmation (your original entry-ready trigger)
    close_above_fast = bool(sma_fast_now is not None and close_now > sma_fast_now)
    rebound_bars = int(max(3, rebound_bars))
    recent_closes = close.tail(rebound_bars).values
    net_up = bool(len(recent_closes) >= 2 and recent_closes[-1] > recent_closes[0])
    rebound_ok = bool(close_above_fast and net_up)

    # HH/HL uptrend logic
    uptrend_info = compute_uptrend_hh_hl(
        df=df,
        window_min=uptrend_window_min,
        window_max=uptrend_window_max,
        pivot_strength=pivot_strength,
        min_hh=min_hh,
        min_hl=min_hl,
    )
    uptrend_actionable = bool(uptrend_info["uptrend_ok"] and trend_ok)

    # NEW: Watchlist = “in correction and holding support, but rebound not started yet”
    watchlist_correction = bool(
        trend_ok
        and uptrend_info["uptrend_ok"]
        and had_correction
        and support_held
        and correction_age_ok
        and (not rebound_ok)
    )

    # Original correction-entry actionable (now also require correction_age_ok, so it matches your screenshot style)
    actionable_correction = bool(trend_ok and had_correction and support_held and correction_age_ok and rebound_ok)

    # FINAL actionable: entry-ready correction OR pure uptrend
    actionable = bool(actionable_correction or uptrend_actionable)

    dist_to_support = None
    if support_floor and support_floor > 0:
        dist_to_support = abs(close_now - support_floor) / support_floor

    resistance_pos = int(idx_high_pos) if idx_high_pos is not None else None

    return {
        "close": close_now,
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "sma_fast_now": safe_float(sma_fast_now),
        "sma_slow_now": safe_float(sma_slow_now),

        "trend_ok": bool(trend_ok),

        # Resistance (most recent swing high pivot)
        "recent_swing_high": safe_float(float(recent_high)),
        "recent_swing_high_date": str(df.index[resistance_pos].date()) if resistance_pos is not None else None,
        "resistance_level": safe_float(float(recent_high)),

        "pullback_from_swing_high_pct": safe_float(pullback_pct),
        "had_correction": bool(had_correction),

        # Correction age
        "bars_since_swing_high": int(bars_since_swing_high),
        "correction_age_ok": bool(correction_age_ok),

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

        # HH/HL structure
        **uptrend_info,
        "uptrend_actionable": bool(uptrend_actionable),

        # NEW: watchlist vs entry
        "watchlist_correction": bool(watchlist_correction),

        # final flag (existing name)
        "actionable_after_correction": bool(actionable),

        "dist_to_support": safe_float(dist_to_support),
        "actionable_reason": (
            "watchlist_correction" if watchlist_correction
            else ("correction_setup" if actionable_correction else ("hh_hl_uptrend" if uptrend_actionable else "none"))
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
    corr_bars_min: int,
    corr_bars_max: int,
    only_actionable: bool,
    only_watchlist: bool,
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

        # Ensure enough bars for SMA + pivots
        min_len = max(
            140,
            int(sma_slow) + 10,
            int(uptrend_window_max) + int(pivot_strength) * 2 + 10,
            int(swing_window) + int(pivot_strength) * 2 + 10,
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
            corr_bars_min=corr_bars_min,
            corr_bars_max=corr_bars_max,
        )

        if only_watchlist and not info["watchlist_correction"]:
            continue

        if only_actionable and not info["actionable_after_correction"]:
            continue

        # Ranking: prefer watchlist or actionable + closer to resistance + strong structure
        score = 0.0
        if info["watchlist_correction"]:
            score += 1.0
        if info["actionable_after_correction"]:
            score += 0.6

        pb = info.get("pullback_from_swing_high_pct") or 0.0
        # moderate pullbacks better than tiny ones
        score += min(float(pb) / 10.0, 0.8)

        d = info.get("dist_to_support")
        if d is not None:
            score += max(0.0, 0.5 - float(d))

        score += min(0.3, 0.05 * float(info.get("uptrend_hh_count", 0)))
        score += min(0.3, 0.05 * float(info.get("uptrend_hl_count", 0)))

        results.append({
            "ticker": t,
            "tf": tf,
            "bars_used": int(len(df)),
            **info,

            # price quality metadata (NO filtering)
            "priceQuality": pq,
            "priceQualityOk": pq["price_quality_ok"],
            "avgVolume50": pq.get("avg_volume_50"),

            "rank_score": float(score),
        })


    results.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)

    # one row per ticker
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in results:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        out.append(r)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correction + Watchlist Scanner (1D) -> JSON")
    p.add_argument("--tickers", nargs="*", default=[], help="Tickers list")
    p.add_argument("--tickers_file", default="", help="One ticker per line")

    p.add_argument("--tf", default="1d", help="Must be 1d")
    p.add_argument("--lookback_days", type=int, default=520)
    p.add_argument("--max_bars", type=int, default=320)

    p.add_argument("--sma_fast", type=int, default=20)
    p.add_argument("--sma_slow", type=int, default=50)

    p.add_argument("--min_pullback_pct", type=float, default=3.0)

    # swing_window = lookback to find MOST RECENT swing high pivot
    p.add_argument("--swing_window", type=int, default=80)

    # support_lookback = lookback before swing high to find MOST RECENT pivot low
    p.add_argument("--support_lookback", type=int, default=80)

    p.add_argument("--rebound_bars", type=int, default=4)
    p.add_argument("--support_tolerance_pct", type=float, default=1.0)

    # uptrend structure
    p.add_argument("--uptrend_window_min", type=int, default=30)
    p.add_argument("--uptrend_window_max", type=int, default=50)
    p.add_argument("--pivot_strength", type=int, default=2)
    p.add_argument("--min_hh", type=int, default=3)
    p.add_argument("--min_hl", type=int, default=3)

    # NEW: correction age window (bars since swing high)
    p.add_argument("--corr_bars_min", type=int, default=3)
    p.add_argument("--corr_bars_max", type=int, default=12)

    # output filters
    p.add_argument("--only_actionable", action="store_true", help="Only entry-ready (rebound confirmed OR pure uptrend)")
    p.add_argument("--only_watchlist", action="store_true", help="Only watchlist-style corrections (waiting)")

    p.add_argument("--out", default="watchlist_output.json")
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
            "corr_bars_min": args.corr_bars_min,
            "corr_bars_max": args.corr_bars_max,
            "only_actionable": bool(args.only_actionable),
            "only_watchlist": bool(args.only_watchlist),
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
        corr_bars_min=args.corr_bars_min,
        corr_bars_max=args.corr_bars_max,
        only_actionable=bool(args.only_actionable),
        only_watchlist=bool(args.only_watchlist),
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
                "actionable_reason",
                "watchlist_correction",
                "actionable_after_correction",
                "bars_since_swing_high",
                "pullback_from_swing_high_pct",
                "support_held",
                "rebound_ok",
                "uptrend_hh_count",
                "uptrend_hl_count",
                "rank_score",
            ]
        ].head(30)
        print(dfp.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
