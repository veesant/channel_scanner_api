#!/usr/bin/env python3
"""
Reversal Scanner (1D): Prior Downtrend -> Fresh Uptrend (last 1–2 weeks) -> Optional Correction Tag

Find stocks that:
1) Were in a prior downtrend over the last N_DOWN bars (default 50)
2) Show a clear uptrend over the last N_UP bars (default 10) with HH + HL structure
3) Optionally: if the last N_CORR bars (default 8) are a pullback / consolidation below recent resistance,
   tag the stock as state="correction" (otherwise state="uptrend").

This is meant to match: downtrend -> sharp reversal -> new uptrend 1–2 weeks -> possible pullback.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Data utils
# -----------------------------

def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    need = {"Open", "High", "Low", "Close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df[["Open", "High", "Low", "Close"]].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df.sort_index().dropna()


def download_daily(ticker: str, lookback_days: int, threads: bool = True) -> pd.DataFrame:
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

            # Exponential backoff + jitter
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


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def linreg_slope(y: np.ndarray) -> float:
    """Slope of y vs index using least squares."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or not np.all(np.isfinite(y)):
        return 0.0
    x = np.arange(len(y), dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom != 0 else 0.0


# -----------------------------
# Pivot / HH-HL utils
# -----------------------------

def pivot_points(values: np.ndarray, left: int, right: int, mode: str) -> List[Tuple[int, float]]:
    """
    Pivot detector (unique extrema):
      pivot high: values[i] is unique max in [i-left, i+right]
      pivot low : values[i] is unique min in [i-left, i+right]
    """
    n = len(values)
    if n == 0:
        return []
    left = max(1, int(left))
    right = max(1, int(right))

    piv: List[Tuple[int, float]] = []
    for i in range(left, n - right):
        w = values[i - left : i + right + 1]
        v = values[i]
        if not np.isfinite(v):
            continue
        if mode == "high":
            m = np.nanmax(w)
            if np.isfinite(m) and v == m and np.sum(w == m) == 1:
                piv.append((i, float(v)))
        elif mode == "low":
            m = np.nanmin(w)
            if np.isfinite(m) and v == m and np.sum(w == m) == 1:
                piv.append((i, float(v)))
        else:
            raise ValueError("mode must be 'high' or 'low'")
    return piv


def count_hh_hl(highs: np.ndarray, lows: np.ndarray, pivot_strength: int) -> Dict[str, Any]:
    ph = pivot_points(highs, pivot_strength, pivot_strength, "high")
    pl = pivot_points(lows, pivot_strength, pivot_strength, "low")

    hh = 0
    prev = None
    for _, v in ph:
        if prev is None:
            prev = v
            continue
        if v > prev:
            hh += 1
        prev = v

    hl = 0
    prev = None
    for _, v in pl:
        if prev is None:
            prev = v
            continue
        if v > prev:
            hl += 1
        prev = v

    return {
        "pivot_highs": len(ph),
        "pivot_lows": len(pl),
        "hh_count": hh,
        "hl_count": hl,
        "ph": ph,
        "pl": pl,
    }


def most_recent_pivot_high(highs: np.ndarray, pivot_strength: int) -> Optional[Tuple[int, float]]:
    ph = pivot_points(highs, pivot_strength, pivot_strength, "high")
    return ph[-1] if ph else None


# -----------------------------
# Strategy config & evaluation
# -----------------------------

@dataclass
class Config:
    # Windows
    n_down: int = 50          # prior downtrend window
    n_up: int = 10            # last 1–2 weeks window
    n_corr: int = 8           # correction window to tag

    # Trend filters
    sma_slow: int = 50
    sma_fast: int = 20

    # HH/HL strictness for the *short* uptrend window
    pivot_strength: int = 1   # 1 is more sensitive (better for 7–12 bar moves); use 2 for stricter swings
    min_hh_steps: int = 1
    min_hl_steps: int = 1

    # Correction (tagging) rules (percent from resistance)
    corr_min_pct: float = 0.5
    corr_max_pct: float = 8.0
    near_resistance_pct: float = 4.0

    # Safety: avoid “breakout already happened” in correction window
    breakout_wiggle_pct: float = 0.2


def evaluate_ticker(df: pd.DataFrame, cfg: Config) -> Optional[Dict[str, Any]]:
    need = cfg.n_down + cfg.n_up
    if df is None or df.empty or len(df) < max(need + cfg.n_corr + 10, cfg.sma_slow + 20):
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    close_now = float(close.iloc[-1])
    sma50 = safe_float(sma(close, cfg.sma_slow).iloc[-1])
    sma20 = safe_float(sma(close, cfg.sma_fast).iloc[-1])

    # Trend sanity: allow reversals that are reclaiming SMA20; require at least above SMA50 OR very close to it
    trend_ok = bool(sma50 is not None and close_now >= sma50 * 0.995)

    # --- Prior downtrend window (N_DOWN bars just before the uptrend window) ---
    # Use the segment that ends right before the last N_UP bars.
    seg_down = df.iloc[-(cfg.n_down + cfg.n_up) : -cfg.n_up]
    down_closes = seg_down["Close"].values.astype(float)

    down_slope = linreg_slope(down_closes)
    down_ok = bool(down_slope < 0 and down_closes[-1] < down_closes[0])

    # Optional stronger downtrend filter: ended below SMA50 at that time
    # (if SMA50 not available then ignore)
    if len(seg_down) >= cfg.sma_slow:
        sma50_then = safe_float(sma(seg_down["Close"], cfg.sma_slow).iloc[-1])
        if sma50_then is not None:
            down_ok = bool(down_ok and down_closes[-1] < sma50_then)

    # --- Fresh uptrend window (last N_UP bars) ---
    seg_up = df.tail(cfg.n_up)
    up_closes = seg_up["Close"].values.astype(float)
    up_highs = seg_up["High"].values.astype(float)
    up_lows = seg_up["Low"].values.astype(float)

    up_slope = linreg_slope(up_closes)
    hhhl = count_hh_hl(up_highs, up_lows, cfg.pivot_strength)

    up_ok = bool(
        up_slope > 0
        and up_closes[-1] > up_closes[0]
        and (sma20 is None or close_now > sma20)  # reclaim fast MA helps match your images
        and hhhl["hh_count"] >= cfg.min_hh_steps
        and hhhl["hl_count"] >= cfg.min_hl_steps
    )

    if not (down_ok and up_ok and trend_ok):
        return None

    # --- Define resistance: most recent pivot high in the uptrend window ---
    piv = most_recent_pivot_high(up_highs, cfg.pivot_strength)
    if piv is None:
        resistance = float(np.nanmax(up_highs))
        res_local_idx = int(np.nanargmax(up_highs))
    else:
        res_local_idx, resistance = int(piv[0]), float(piv[1])

    # --- Optional correction tagging (last N_CORR bars) ---
    seg_corr = df.tail(cfg.n_corr)
    corr_highs = seg_corr["High"].values.astype(float)
    corr_lows = seg_corr["Low"].values.astype(float)

    pullback_pct = (resistance - close_now) / resistance * 100.0 if resistance > 0 else 0.0
    below_res = close_now < resistance

    pullback_ok = bool(cfg.corr_min_pct <= pullback_pct <= cfg.corr_max_pct)
    near_res = bool(pullback_pct <= cfg.near_resistance_pct)

    max_corr_high = float(np.nanmax(corr_highs))
    wiggle = 1.0 + (cfg.breakout_wiggle_pct / 100.0)
    no_breakout_yet = bool(max_corr_high <= resistance * wiggle)

    in_correction = bool(below_res and pullback_ok and near_res and no_breakout_yet)

    state = "correction" if in_correction else "uptrend"

    # Rank: prefer correction setups close to breakout + stronger HH/HL + stronger up slope
    score = 0.0
    score += 1.0
    score += 0.6 if state == "correction" else 0.0
    score += max(0.0, (cfg.near_resistance_pct - pullback_pct) / max(0.01, cfg.near_resistance_pct))
    score += 0.15 * float(hhhl["hh_count"])
    score += 0.10 * float(hhhl["hl_count"])
    score += min(0.3, max(0.0, up_slope / (np.nanstd(up_closes) + 1e-6)))  # normalized slope bonus

    return {
        "close": safe_float(close_now),
        "sma20": safe_float(sma20),
        "sma50": safe_float(sma50),
        "trend_ok": bool(trend_ok),

        "prior_downtrend_ok": bool(down_ok),
        "down_slope": safe_float(down_slope),

        "fresh_uptrend_ok": bool(up_ok),
        "up_slope": safe_float(up_slope),
        "up_hh_steps": int(hhhl["hh_count"]),
        "up_hl_steps": int(hhhl["hl_count"]),
        "up_pivot_highs": int(hhhl["pivot_highs"]),
        "up_pivot_lows": int(hhhl["pivot_lows"]),

        "resistance": safe_float(resistance),
        "pullback_from_resistance_pct": safe_float(pullback_pct),

        "state": state,  # "uptrend" or "correction"
        "rank_score": float(score),
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reversal Scanner (1D): Downtrend -> Fresh Uptrend -> Optional Correction")
    p.add_argument("--tickers", nargs="*", default=[], help="Tickers list")
    p.add_argument("--tickers_file", default="", help="One ticker per line")

    p.add_argument("--out", default="reversal_scan.json")
    p.add_argument("--lookback_days", type=int, default=520)
    p.add_argument("--max_bars", type=int, default=350)
    p.add_argument("--no_threads", action="store_true")

    # Windows
    p.add_argument("--n_down", type=int, default=50)
    p.add_argument("--n_up", type=int, default=10)
    p.add_argument("--n_corr", type=int, default=8)

    # Filters
    p.add_argument("--sma_slow", type=int, default=50)
    p.add_argument("--sma_fast", type=int, default=20)

    # HH/HL
    p.add_argument("--pivot_strength", type=int, default=1)
    p.add_argument("--min_hh_steps", type=int, default=1)
    p.add_argument("--min_hl_steps", type=int, default=1)

    # Correction tag thresholds
    p.add_argument("--corr_min_pct", type=float, default=0.5)
    p.add_argument("--corr_max_pct", type=float, default=8.0)
    p.add_argument("--near_resistance_pct", type=float, default=4.0)
    p.add_argument("--breakout_wiggle_pct", type=float, default=0.2)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers.extend(args.tickers or [])
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]

    cfg = Config(
        n_down=args.n_down,
        n_up=args.n_up,
        n_corr=args.n_corr,
        sma_slow=args.sma_slow,
        sma_fast=args.sma_fast,
        pivot_strength=args.pivot_strength,
        min_hh_steps=args.min_hh_steps,
        min_hl_steps=args.min_hl_steps,
        corr_min_pct=args.corr_min_pct,
        corr_max_pct=args.corr_max_pct,
        near_resistance_pct=args.near_resistance_pct,
        breakout_wiggle_pct=args.breakout_wiggle_pct,
    )

    payload: Dict[str, Any] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "tf": "1d",
        "params": asdict(cfg),
        "count": 0,
        "data": [],
    }

    if not tickers:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("No tickers provided.")
        return 2

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        df = download_daily(t, lookback_days=args.lookback_days, threads=not args.no_threads)
        if df.empty:
            continue
        df = df.tail(int(args.max_bars)).copy()

        info = evaluate_ticker(df, cfg)
        if info is None:
            continue

        rows.append({"ticker": t, "tf": "1d", "bars_used": int(len(df)), **info})

    rows.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)
    payload["count"] = len(rows)
    payload["data"] = rows

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(rows)} matches to {args.out}")
    if rows:
        show = pd.DataFrame(rows)[
            ["ticker", "state", "close", "resistance", "pullback_from_resistance_pct", "up_hh_steps", "up_hl_steps", "rank_score"]
        ].head(25)
        print(show.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
