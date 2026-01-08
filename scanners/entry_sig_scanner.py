#!/usr/bin/env python3
"""
Clean HH/HL Uptrend Scanner (DAILY)

Goal (matches your 2nd chart):
- In the last N bars (default 35), price shows a continuous uptrend:
  - swing structure alternates L/H/L/H...
  - ALL swing highs are higher than prior swing highs (HH chain)
  - ALL swing lows are higher than prior swing lows (HL chain)
  - Trend filter: SMA fast > SMA slow and last close > SMA slow
  - Optional strictness: max drawdown within window must be below a cap
  - Positive slope of closes over the window

Inputs:
- tickers_file: one ticker per line (blank lines + # comments ignored)
- Uses yfinance daily data with auto_adjust=True

Output:
- JSON file (strict JSON, allow_nan=False) with records + metadata
- Console preview table

Install:
  pip install yfinance pandas numpy
Run:
  python hh_hl_scanner_clean.py --tickers_file data/tickers.txt --out output/hh_hl.json
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
# Helpers
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
    Recursively convert:
      - NaN -> None
      - +inf/-inf -> None
      - numpy scalars -> python scalars
      - pandas Timestamp -> ISO string
    so json.dump(allow_nan=False) succeeds.
    """
    if obj is None:
        return None

    # pandas timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    # numpy scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v

    # python float
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj

    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_sanitize(v) for v in obj]

    return obj


def extract_ticker_df(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """
    Robust extraction for yfinance multi-ticker downloads.
    Handles MultiIndex columns or single ticker DataFrame.
    """
    if data is None or data.empty:
        return None

    # Multi-ticker -> MultiIndex columns: (field, ticker)
    if isinstance(data.columns, pd.MultiIndex):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        cols = []
        for field in fields:
            if (field, ticker) in data.columns:
                cols.append((field, ticker))
        if not cols:
            return None
        df = data.loc[:, cols].copy()
        df.columns = [c[0] for c in df.columns]  # flatten
        return df

    # Single ticker dataframe: columns are normal
    expected = {"Open", "High", "Low", "Close"}
    if not expected.issubset(set(data.columns)):
        return None
    return data.copy()


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (float, int, np.floating, np.integer)):
            if np.isnan(x):
                return None
            return float(x)
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


# -----------------------------
# Pivot logic (same idea as your original)
# -----------------------------

def find_pivot_highs(high: pd.Series, n: int) -> List[pd.Timestamp]:
    """
    Pivot high at i if high[i] is the strictly-unique max in [i-n, i+n].
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
        # strictly unique max
        if vals[i] == m and np.sum(window == m) == 1:
            idxs.append(high.index[i])
    return idxs


def find_pivot_lows(low: pd.Series, n: int) -> List[pd.Timestamp]:
    """
    Pivot low at i if low[i] is the strictly-unique min in [i-n, i+n].
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
        # strictly unique min
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
      - merged, time-ordered pivot sequence with alternation enforced (compressed)
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

    # Compress consecutive same-kind pivots (noise):
    # - For consecutive highs, keep the higher one
    # - For consecutive lows, keep the lower one
    compressed: List[PivotPoint] = []
    for p in pts:
        if not compressed:
            compressed.append(p)
            continue
        last = compressed[-1]
        if p.kind != last.kind:
            compressed.append(p)
            continue

        # same kind -> keep the more "extreme" pivot
        if p.kind == "H":
            if p.price > last.price:
                compressed[-1] = p
        else:  # "L"
            if p.price < last.price:
                compressed[-1] = p

    return compressed, highs, lows


# -----------------------------
# "Clean uptrend" logic (matches your desired pattern)
# -----------------------------

def linear_slope(values: np.ndarray) -> float:
    """
    Returns slope of best-fit line y = a*x + b.
    """
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    # handle NaNs
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    a, _b = np.polyfit(x, y, 1)
    return float(a)


def max_drawdown_pct(close: np.ndarray) -> float:
    """
    Max drawdown % over the series, in percent (positive number).
    """
    if len(close) < 2:
        return 0.0
    c = close.astype(float)
    # handle NaNs
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
    """
    Returns (passes, debug_metrics)
    """
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

    # pivot points that fall inside the window
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

    # Need alternation and a meaningful sequence length
    # (At least 2*min_pivots_per_side points is a good minimum)
    if len(pts) < 2 * min_pivots_per_side:
        return False, {
            "structure_points": float(len(pts)),
        }

    # Verify alternation in the pivot stream (after compression already)
    for i in range(1, len(pts)):
        if pts[i].kind == pts[i - 1].kind:
            return False, {"pivot_alternation_ok": 0.0}

    # Verify HH chain: every successive pivot high is higher
    high_prices = [p.price for p in highs]
    hh_chain_ok = all(high_prices[i] > high_prices[i - 1] for i in range(1, len(high_prices)))

    # Verify HL chain: every successive pivot low is higher
    low_prices = [p.price for p in lows]
    hl_chain_ok = all(low_prices[i] > low_prices[i - 1] for i in range(1, len(low_prices)))

    if not (hh_chain_ok and hl_chain_ok):
        return False, {
            "hh_chain_ok": 1.0 if hh_chain_ok else 0.0,
            "hl_chain_ok": 1.0 if hl_chain_ok else 0.0,
        }

    # Trend filter (strongly recommended for "no downtrend in this period")
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

    # Positive slope of closes across the whole window
    slope_ok = True
    slope = linear_slope(window["Close"].values)
    if require_positive_slope:
        slope_ok = slope > 0
        if not slope_ok:
            return False, {"close_slope": slope, "slope_ok": 0.0}

    # Optional: cap max drawdown inside the window (filters "downtrend legs")
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
# Main scan per ticker
# -----------------------------

def scan_one(
    ticker: str,
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> Optional[Dict]:
    df = extract_ticker_df(data, ticker)
    if df is None or df.empty:
        return None

    # Drop rows with missing core OHLC
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if df.empty:
        return None

    # Restrict to last max_trading_days
    df = df.tail(args.max_trading_days).copy()
    if len(df) < max(10, args.structure_bars // 2):
        return None

    # Pivots (on trimmed df)
    pivot_seq, raw_highs, raw_lows = build_pivot_sequence(df, args.pivot)

    # Legacy HH/HL check (kept for reference)
    has_hh = False
    has_hl = False
    last_high_1 = last_high_2 = None
    last_low_1 = last_low_2 = None

    if len(raw_highs) >= 2:
        h1, h2 = raw_highs[-2], raw_highs[-1]
        last_high_1 = safe_float(df.at[h1, "High"])
        last_high_2 = safe_float(df.at[h2, "High"])
        if last_high_1 is not None and last_high_2 is not None:
            has_hh = last_high_2 > last_high_1

    if len(raw_lows) >= 2:
        l1, l2 = raw_lows[-2], raw_lows[-1]
        last_low_1 = safe_float(df.at[l1, "Low"])
        last_low_2 = safe_float(df.at[l2, "Low"])
        if last_low_1 is not None and last_low_2 is not None:
            has_hl = last_low_2 > last_low_1

    legacy_has_hh_hl = bool(has_hh and has_hl)

    # Clean uptrend gate (your desired pattern)
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

    # SMA flags for output
    df["SMA_FAST"] = df["Close"].rolling(args.sma_fast).mean()
    df["SMA_SLOW"] = df["Close"].rolling(args.sma_slow).mean()
    last_close = safe_float(df["Close"].iloc[-1])
    sma_fast_last = safe_float(df["SMA_FAST"].iloc[-1])
    sma_slow_last = safe_float(df["SMA_SLOW"].iloc[-1])

    sma_fast_gt_slow = None
    if sma_fast_last is not None and sma_slow_last is not None:
        sma_fast_gt_slow = bool(sma_fast_last > sma_slow_last)

    # Correction fields (same spirit as your original)
    pullback_pct = None
    above_support_pct = None
    correction_started = False
    correction_stage = "none"
    near_support = False
    near_breakout = False

    if last_close is not None and last_high_2 is not None and last_low_2 is not None:
        if last_high_2 > 0:
            pullback_pct = (last_high_2 - last_close) / last_high_2 * 100.0
        if last_low_2 > 0:
            above_support_pct = (last_close - last_low_2) / last_low_2 * 100.0

        if last_low_2 <= last_close <= last_high_2:
            correction_started = True

        if pullback_pct is not None:
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

        if above_support_pct is not None:
            near_support = bool(above_support_pct <= args.near_support_pct)

        if last_high_2 is not None and last_close is not None and last_high_2 > 0:
            dist_to_breakout_pct = (last_high_2 - last_close) / last_high_2 * 100.0
            near_breakout = bool(dist_to_breakout_pct <= args.near_breakout_pct)

    # Output record
    last_date = str(df.index[-1].date()) if len(df.index) else None

    return {
        "ticker": ticker,
        # This is now the "clean pattern" flag you want:
        "has_hh_hl": bool(clean_ok),

        # keep legacy flags for debugging / comparison
        "legacy_has_hh_hl": bool(legacy_has_hh_hl),
        "legacy_has_hh": bool(has_hh),
        "legacy_has_hl": bool(has_hl),

        "last_date": last_date,
        "last_close": last_close,

        "last_high_1": last_high_1,
        "last_high_2": last_high_2,
        "last_low_1": last_low_1,
        "last_low_2": last_low_2,

        "pullback_pct": pullback_pct,
        "above_support_pct": above_support_pct,
        "correction_started": bool(correction_started),
        "correction_stage": correction_stage,
        "near_support": bool(near_support),
        "near_breakout": bool(near_breakout),

        "sma_fast": args.sma_fast,
        "sma_slow": args.sma_slow,
        "sma_fast_last": sma_fast_last,
        "sma_slow_last": sma_slow_last,
        "sma_fast_gt_slow": sma_fast_gt_slow,

        # Debug metrics explaining why it passed/failed your "clean trend" definition
        "structure_bars": args.structure_bars,
        "min_pivots_per_side": args.min_pivots_per_side,
        **{f"metric_{k}": v for k, v in metrics.items()},
    }


# -----------------------------
# CLI / Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean HH/HL Uptrend Scanner (DAILY)")

    p.add_argument("--tickers_file", default="nifty.txt", help="Text file with tickers, one per line")
    p.add_argument("--out", default="output/hh_hl.json", help="Output JSON file path")

    # Download window
    p.add_argument("--last_days", type=int, default=80, help="Calendar days to download (bigger helps for pivots)")
    p.add_argument("--max_trading_days", type=int, default=60, help="Keep last N trading days after download")

    # Pivot detection
    p.add_argument("--pivot", type=int, default=3, help="Pivot window n: uses [i-n..i+n]. Try 3 for cleaner swings")

    # Clean structure requirements (your pattern)
    p.add_argument("--structure_bars", type=int, default=35, help="Bars to evaluate 'continuous uptrend' structure")
    p.add_argument("--min_pivots_per_side", type=int, default=3, help="Min pivot highs and pivot lows within structure window")

    # Trend filters
    p.add_argument("--sma_fast", type=int, default=10, help="Fast SMA")
    p.add_argument("--sma_slow", type=int, default=20, help="Slow SMA")
    p.add_argument("--disable_sma_gate", action="store_true", help="Disable SMA gate (fast>slow and close>slow)")
    p.add_argument("--disable_slope_gate", action="store_true", help="Disable positive close slope requirement")

    # Strictness: max drawdown cap in last structure_bars (set <=0 to disable)
    p.add_argument("--max_drawdown_pct", type=float, default=12.0, help="Max drawdown %% allowed in structure window (<=0 disables)")

    # Near flags
    p.add_argument("--near_support_pct", type=float, default=3.0, help="Near support if within this %% above last pivot low")
    p.add_argument("--near_breakout_pct", type=float, default=2.0, help="Near breakout if within this %% below last pivot high")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    tickers = read_tickers(args.tickers_file)
    if not tickers:
        print("No tickers found in tickers file.", file=sys.stderr)
        return 2

    # Download (batch)
    data = yf.download(
        tickers=tickers,
        period=None,
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
            # Keep going; record error row (optional)
            rows.append({"ticker": t, "has_hh_hl": False, "error": str(e)})

    df = pd.DataFrame(rows)

    # Sort: clean trend first, then corrections/near flags
    if not df.empty and "has_hh_hl" in df.columns:
        df["has_hh_hl"] = df["has_hh_hl"].fillna(False).astype(bool)
        if "correction_started" in df.columns:
            df["correction_started"] = df["correction_started"].fillna(False).astype(bool)
        if "near_support" in df.columns:
            df["near_support"] = df["near_support"].fillna(False).astype(bool)
        if "near_breakout" in df.columns:
            df["near_breakout"] = df["near_breakout"].fillna(False).astype(bool)

        sort_cols = ["has_hh_hl", "correction_started", "near_support", "near_breakout", "last_date"]
        ascending = [False, False, False, False, False]
        existing = [c for c in sort_cols if c in df.columns]
        if existing:
            df = df.sort_values(by=existing, ascending=ascending[: len(existing)], kind="mergesort")

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
            "near_support_pct": args.near_support_pct,
            "near_breakout_pct": args.near_breakout_pct,
        },
        "count": int(len(df)),
        "count_hhhl_clean": int(df["has_hh_hl"].sum()) if (not df.empty and "has_hh_hl" in df.columns) else 0,
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

    # Preview
    preview_cols = [
        "ticker", "has_hh_hl",
        "legacy_has_hh_hl",
        "metric_structure_high_pivots", "metric_structure_low_pivots",
        "metric_close_slope", "metric_max_drawdown_pct",
        "correction_stage", "pullback_pct", "above_support_pct",
        "near_support", "near_breakout",
        "sma_fast_gt_slow", "last_close",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print("\nPreview (top 25):")
    print(df[preview_cols].head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
