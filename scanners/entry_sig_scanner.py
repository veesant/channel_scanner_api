#!/usr/bin/env python3
"""
Entry Signal Scanner (DAILY) - Clean HH/HL + Pullback-to-HL + Coil-under-HH

What it detects (your circled setup):
1) Clean uptrend structure over last N bars:
   - pivots alternate L/H/L/H...
   - all pivot highs rising (HH chain)
   - all pivot lows rising (HL chain)
2) Current price is near the latest HL (pullback holding support)
3) Current price is not far below latest HH (coiled for breakout)

Output:
- JSON file with fields:
  - has_hh_hl: clean uptrend pass/fail
  - entry_signal: TRUE = your desired “at HL with breakout potential” setup
  - at_hl_zone, coiled_below_hh: explain why entry_signal is true
  - support_distance_pct, breakout_distance_pct: numeric distances

Install:
  pip install yfinance pandas numpy

Run:
  python entry_signal_scanner.py --tickers_file data/tickers.txt --out output/entry_signals.json
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
    """
    Make payload strict-JSON compliant (no NaN/inf, convert numpy/pandas scalars).
    """
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
    """
    Robust extraction for yfinance multi-ticker downloads.
    """
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
    """
    Pivot high at i if high[i] is strictly-unique max in [i-n, i+n]
    """
    idxs: List[pd.Timestamp] = []
    if len(high) < 2 * n + 1:
        return idxs

    vals = high.values
    for i in range(n, len(high) - n):
        window = vals[i - n : i + n + 1]
        m = np.nanmax(window)
        if np.isnan(m):
            continue
        if vals[i] == m and np.sum(window == m) == 1:
            idxs.append(high.index[i])
    return idxs


def find_pivot_lows(low: pd.Series, n: int) -> List[pd.Timestamp]:
    """
    Pivot low at i if low[i] is strictly-unique min in [i-n, i+n]
    """
    idxs: List[pd.Timestamp] = []
    if len(low) < 2 * n + 1:
        return idxs

    vals = low.values
    for i in range(n, len(low) - n):
        window = vals[i - n : i + n + 1]
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


def build_pivot_sequence(
    df: pd.DataFrame,
    pivot_n: int
) -> Tuple[List[PivotPoint], List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Returns:
      - compressed, time-ordered pivot sequence (enforces alternation by compressing same-kind)
      - raw pivot high timestamps
      - raw pivot low timestamps
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

    # Compress consecutive same-kind pivots (noise)
    compressed: List[PivotPoint] = []
    for p in pts:
        if not compressed:
            compressed.append(p)
            continue
        last = compressed[-1]
        if p.kind != last.kind:
            compressed.append(p)
            continue

        # same kind: keep more extreme
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
    x = x[mask]
    y = y[mask]
    a, _b = np.polyfit(x, y, 1)
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
# Clean HH/HL uptrend definition
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

    if len(pts) < 2 * min_pivots_per_side:
        return False, {"structure_points": float(len(pts))}

    # alternation check
    for i in range(1, len(pts)):
        if pts[i].kind == pts[i - 1].kind:
            return False, {"pivot_alternation_ok": 0.0}

    # HH chain
    high_prices = [p.price for p in highs]
    hh_chain_ok = all(high_prices[i] > high_prices[i - 1] for i in range(1, len(high_prices)))

    # HL chain
    low_prices = [p.price for p in lows]
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
# Entry signal logic (your circled setup)
# -----------------------------

def compute_entry_signal(
    last_close: Optional[float],
    last_hh: Optional[float],
    last_hl: Optional[float],
    min_above_hl_pct: float,
    max_above_hl_pct: float,
    max_below_hh_pct: float,
) -> Tuple[bool, Optional[float], Optional[float], bool, bool]:
    """
    Returns:
      entry_signal, support_distance_pct, breakout_distance_pct, at_hl_zone, coiled_below_hh
    """
    if last_close is None or last_hh is None or last_hl is None or last_hl <= 0 or last_hh <= 0:
        return False, None, None, False, False

    # how far above HL we are
    support_distance_pct = (last_close - last_hl) / last_hl * 100.0

    # how far below HH we are (negative = already above HH)
    breakout_distance_pct = (last_hh - last_close) / last_hh * 100.0

    at_hl_zone = (support_distance_pct >= min_above_hl_pct) and (support_distance_pct <= max_above_hl_pct)
    coiled_below_hh = (breakout_distance_pct > 0) and (breakout_distance_pct <= max_below_hh_pct)

    entry_signal = bool(at_hl_zone and coiled_below_hh)
    return entry_signal, support_distance_pct, breakout_distance_pct, at_hl_zone, coiled_below_hh


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

    # Keep last max_trading_days
    df = df.tail(args.max_trading_days).copy()
    if len(df) < max(10, args.structure_bars // 2):
        return None

    # pivots
    pivot_seq, raw_highs, raw_lows = build_pivot_sequence(df, args.pivot)

    # Last two raw pivots (for last HH/HL levels)
    last_hh = None
    last_hl = None

    if len(raw_highs) >= 1:
        last_hh = safe_float(df.at[raw_highs[-1], "High"])
    if len(raw_lows) >= 1:
        last_hl = safe_float(df.at[raw_lows[-1], "Low"])

    # clean trend gate
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

    # SMA last values
    df["SMA_FAST"] = df["Close"].rolling(args.sma_fast).mean()
    df["SMA_SLOW"] = df["Close"].rolling(args.sma_slow).mean()
    last_close = safe_float(df["Close"].iloc[-1])
    sma_fast_last = safe_float(df["SMA_FAST"].iloc[-1])
    sma_slow_last = safe_float(df["SMA_SLOW"].iloc[-1])
    sma_fast_gt_slow = None
    if sma_fast_last is not None and sma_slow_last is not None:
        sma_fast_gt_slow = bool(sma_fast_last > sma_slow_last)

    # entry signal (only meaningful if clean trend already true)
    entry_signal = False
    support_distance_pct = None
    breakout_distance_pct = None
    at_hl_zone = False
    coiled_below_hh = False

    if clean_ok:
        entry_signal, support_distance_pct, breakout_distance_pct, at_hl_zone, coiled_below_hh = compute_entry_signal(
            last_close=last_close,
            last_hh=last_hh,
            last_hl=last_hl,
            min_above_hl_pct=args.min_above_hl_pct,
            max_above_hl_pct=args.max_above_hl_pct,
            max_below_hh_pct=args.max_below_hh_pct,
        )

    # basic correction context (optional)
    correction_stage = "none"
    pullback_pct = None
    if last_close is not None and last_hh is not None and last_hh > 0:
        pullback_pct = (last_hh - last_close) / last_hh * 100.0
        if pullback_pct < 0:
            correction_stage = "breakout"
        elif pullback_pct <= 3:
            correction_stage = "early"
        elif pullback_pct <= 7:
            correction_stage = "healthy"
        elif pullback_pct <= 12:
            correction_stage = "deep"
        else:
            correction_stage = "invalid"

    last_date = str(df.index[-1].date()) if len(df.index) else None

    return {
        "ticker": ticker,
        "last_date": last_date,
        "last_close": last_close,

        # Trend qualification
        "has_hh_hl": bool(clean_ok),
        "structure_bars": args.structure_bars,
        "min_pivots_per_side": args.min_pivots_per_side,

        # Levels used for entry decision
        "last_hh": last_hh,
        "last_hl": last_hl,

        # Entry setup flags (THIS is what you filter on)
        "entry_signal": bool(entry_signal),
        "at_hl_zone": bool(at_hl_zone),
        "coiled_below_hh": bool(coiled_below_hh),

        # Distances (for tuning/sorting)
        "support_distance_pct": support_distance_pct,   # % above HL
        "breakout_distance_pct": breakout_distance_pct, # % below HH (>0 means below)

        # Context
        "pullback_pct": pullback_pct,
        "correction_stage": correction_stage,

        # SMA info
        "sma_fast": args.sma_fast,
        "sma_slow": args.sma_slow,
        "sma_fast_last": sma_fast_last,
        "sma_slow_last": sma_slow_last,
        "sma_fast_gt_slow": sma_fast_gt_slow,

        # Debug metrics for why trend passed/failed
        **{f"metric_{k}": v for k, v in metrics.items()},
    }


# -----------------------------
# CLI / Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entry Signal Scanner (Clean HH/HL + Pullback-to-HL + Coil-under-HH)")

    p.add_argument("--tickers_file", default="tickers.txt", help="Text file with tickers, one per line")
    p.add_argument("--out", default="output/entry_signals.json", help="Output JSON file path")

    # download/truncate
    p.add_argument("--last_days", type=int, default=120, help="Calendar days to download")
    p.add_argument("--max_trading_days", type=int, default=80, help="Keep last N trading days after download")

    # pivots & structure window
    p.add_argument("--pivot", type=int, default=3, help="Pivot window n (uses [i-n..i+n])")
    p.add_argument("--structure_bars", type=int, default=35, help="Bars used to validate clean uptrend")
    p.add_argument("--min_pivots_per_side", type=int, default=3, help="Min pivot highs and lows within structure window")

    # trend filters
    p.add_argument("--sma_fast", type=int, default=10, help="Fast SMA")
    p.add_argument("--sma_slow", type=int, default=20, help="Slow SMA")
    p.add_argument("--disable_sma_gate", action="store_true", help="Disable SMA gate (fast>slow and close>slow)")
    p.add_argument("--disable_slope_gate", action="store_true", help="Disable positive close slope requirement")
    p.add_argument("--max_drawdown_pct", type=float, default=12.0, help="Max drawdown %% allowed in structure window (<=0 disables)")

    # entry setup tuning (HL zone + HH proximity)
    p.add_argument("--min_above_hl_pct", type=float, default=0.5, help="Min %% above HL to qualify as 'at HL' (avoid breakdown)")
    p.add_argument("--max_above_hl_pct", type=float, default=4.0, help="Max %% above HL to qualify as 'at HL'")
    p.add_argument("--max_below_hh_pct", type=float, default=5.0, help="Max %% below HH to be considered 'coiled for breakout'")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers = read_tickers(args.tickers_file)
    if not tickers:
        print("No tickers found.", file=sys.stderr)
        return 2

    data = yf.download(
        tickers=tickers,
        start=pd.Timestamp.today(tz="UTC") - pd.Timedelta(days=int(args.last_days)),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )

    rows: List[Dict] = []
    for t in tickers:
        try:
            rec = scan_one(t, data, args)
            if rec is not None:
                rows.append(rec)
        except Exception as e:
            rows.append({"ticker": t, "has_hh_hl": False, "entry_signal": False, "error": str(e)})

    df = pd.DataFrame(rows)

    # sort: entry setups first, then clean trends
    if not df.empty:
        for c in ["entry_signal", "has_hh_hl", "at_hl_zone", "coiled_below_hh"]:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(bool)

        # Prefer smallest breakout_distance_pct (closest to HH) among entry signals
        if "breakout_distance_pct" in df.columns:
            df["breakout_distance_pct"] = pd.to_numeric(df["breakout_distance_pct"], errors="coerce")

        sort_cols = ["entry_signal", "has_hh_hl", "at_hl_zone", "coiled_below_hh", "breakout_distance_pct", "last_date"]
        existing = [c for c in sort_cols if c in df.columns]
        if existing:
            ascending = [False, False, False, False, True, False][: len(existing)]
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
            "min_above_hl_pct": args.min_above_hl_pct,
            "max_above_hl_pct": args.max_above_hl_pct,
            "max_below_hh_pct": args.max_below_hh_pct,
        },
        "count": int(len(df)),
        "count_clean_trend": int(df["has_hh_hl"].sum()) if (not df.empty and "has_hh_hl" in df.columns) else 0,
        "count_entry_signal": int(df["entry_signal"].sum()) if (not df.empty and "entry_signal" in df.columns) else 0,
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

    # quick preview
    preview_cols = [
        "ticker", "entry_signal", "has_hh_hl",
        "support_distance_pct", "breakout_distance_pct",
        "at_hl_zone", "coiled_below_hh",
        "last_close", "last_hl", "last_hh",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("\nPreview (top 25):")
    print(df[preview_cols].head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
