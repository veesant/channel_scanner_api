#!/usr/bin/env python3
"""
HH/HL + Uptrend Pullback + Correction Phase Scanner -> JSON output

Adds:
- "correction_phase" detection (uptrend intact + controlled pullback)
- CLI to filter only correction names: --only_correction

Recommended for TradingView-like patterns: strong impulse up -> pullback/correction

Examples:
  python hh_hl_scanner.py --tickers INTC SERV --tf 4h --only_correction --out output.json
  python hh_hl_scanner.py --tickers_file universe.txt --tf 4h --only_correction --out output.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close"]


def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = []
        for c in df.columns:
            if isinstance(c, tuple) and len(c) > 0:
                flat_cols.append(str(c[0]))
            else:
                flat_cols.append(str(c))
        df = df.copy()
        df.columns = flat_cols

    # Standardize column names
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

    df = df.sort_index().dropna()
    return df


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = normalize_ohlc(df)
    if df.empty:
        return df

    out = (
        df.resample(rule)
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
          .dropna()
    )
    return out


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isfinite(v):
            return v
        return None
    except Exception:
        return None


# ----------------------------
# Pivot detection
# ----------------------------
@dataclass
class PivotInfo:
    last_high_1: Optional[float] = None
    last_high_2: Optional[float] = None
    last_low_1: Optional[float] = None
    last_low_2: Optional[float] = None
    has_hh: bool = False
    has_hl: bool = False
    has_hh_hl: bool = False


def find_pivots(series: pd.Series, left_right: int = 3, mode: str = "high") -> pd.Series:
    s = series.values
    n = len(s)
    piv = np.full(n, np.nan, dtype=float)

    lr = int(left_right)
    if n < (2 * lr + 1):
        return pd.Series(piv, index=series.index)

    for i in range(lr, n - lr):
        window = s[i - lr : i + lr + 1]
        center = s[i]
        if mode == "high":
            if center == np.max(window) and np.sum(window == center) == 1:
                piv[i] = center
        else:
            if center == np.min(window) and np.sum(window == center) == 1:
                piv[i] = center

    return pd.Series(piv, index=series.index)


def extract_last_two(pivot_series: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    vals = pivot_series.dropna().values
    if len(vals) >= 2:
        return float(vals[-1]), float(vals[-2])
    if len(vals) == 1:
        return float(vals[-1]), None
    return None, None


def compute_pivot_info(df: pd.DataFrame, pivot: int = 3) -> PivotInfo:
    ph = find_pivots(df["High"], left_right=pivot, mode="high")
    pl = find_pivots(df["Low"], left_right=pivot, mode="low")

    h1, h2 = extract_last_two(ph)
    l1, l2 = extract_last_two(pl)

    has_hh = (h1 is not None and h2 is not None and h1 > h2)
    has_hl = (l1 is not None and l2 is not None and l1 > l2)

    return PivotInfo(
        last_high_1=h1,
        last_high_2=h2,
        last_low_1=l1,
        last_low_2=l2,
        has_hh=has_hh,
        has_hl=has_hl,
        has_hh_hl=bool(has_hh and has_hl),
    )


# ----------------------------
# Trend & correction logic
# ----------------------------
def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def trend_filters(df: pd.DataFrame, sma_fast: int = 20, sma_slow: int = 50) -> Dict[str, Any]:
    close = df["Close"]
    s_fast = sma(close, sma_fast)
    s_slow = sma(close, sma_slow)

    fast_now = safe_float(s_fast.iloc[-1])
    slow_now = safe_float(s_slow.iloc[-1])

    # fast slope using last ~6 bars
    if len(s_fast.dropna()) >= 6:
        slope = safe_float(s_fast.iloc[-1] - s_fast.iloc[-6])
    else:
        slope = None

    fast_gt_slow = (fast_now is not None and slow_now is not None and fast_now > slow_now)
    fast_slope_up = (slope is not None and slope > 0)

    return {
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "sma_fast_now": fast_now,
        "sma_slow_now": slow_now,
        "sma_fast_gt_slow": bool(fast_gt_slow),
        "sma_fast_slope_up": bool(fast_slope_up),
    }


def linear_slope(values: np.ndarray) -> float:
    """Simple OLS slope for y over x=0..n-1"""
    y = values.astype(float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    m, _b = np.polyfit(x, y, 1)
    return float(m)


def classify_setup(
    df: pd.DataFrame,
    piv: PivotInfo,
    trend: Dict[str, Any],
    pullback_min_pct: float = 2.0,
    correction_min_pct: float = 3.0,
    correction_bars: int = 8,
) -> Dict[str, Any]:
    close_now = float(df["Close"].iloc[-1])

    # Uptrend intact = HL pivots OR MA confirmation
    uptrend_intact = (piv.has_hl or (trend["sma_fast_gt_slow"] and trend["sma_fast_slope_up"]))

    # Pullback from last pivot high
    pullback = False
    pullback_from_high_pct = None
    if piv.last_high_1 is not None and piv.last_high_1 > 0:
        pullback_from_high_pct = (piv.last_high_1 - close_now) / piv.last_high_1 * 100.0
        pullback = bool(pullback_from_high_pct >= pullback_min_pct)

    # Controlled correction = down-drift in recent bars (negative slope) BUT no breakdown below latest pivot low
    correction = False
    corr_slope = None
    if len(df) >= correction_bars:
        recent = df["Close"].iloc[-correction_bars:].values
        corr_slope = linear_slope(recent)  # negative => drifting down
        slope_down = corr_slope < 0

        support_ok = True
        if piv.last_low_1 is not None:
            support_ok = close_now > piv.last_low_1  # still above last pivot low

        correction = bool(
            uptrend_intact
            and pullback_from_high_pct is not None
            and pullback_from_high_pct >= correction_min_pct
            and slope_down
            and support_ok
        )

    # strict HH+HL (fresh uptrend)
    setup_hh_hl = piv.has_hh_hl

    # uptrend pullback (more permissive than HH/HL)
    setup_uptrend_pullback = bool(uptrend_intact and pullback and (piv.last_low_1 is None or close_now > piv.last_low_1))

    # Near support/resistance using pivots
    near_support = False
    near_resistance = False
    support = piv.last_low_1
    resistance = piv.last_high_1

    if support is not None and support > 0:
        near_support = abs(close_now - support) / support <= 0.015

    if resistance is not None and resistance > 0:
        near_resistance = abs(close_now - resistance) / resistance <= 0.015

    if near_support:
        zone = "Near Support"
    elif near_resistance:
        zone = "Near Resistance"
    else:
        zone = "Mid"

    return {
        "close": close_now,
        "uptrend_intact": bool(uptrend_intact),
        "pullback": bool(pullback),
        "pullback_from_high_pct": safe_float(pullback_from_high_pct),
        "setup_hh_hl": bool(setup_hh_hl),
        "setup_uptrend_pullback": bool(setup_uptrend_pullback),
        "correction_phase": bool(correction),
        "correction_slope": safe_float(corr_slope),
        "near_support": bool(near_support),
        "near_resistance": bool(near_resistance),
        "support_level": safe_float(support),
        "resistance_level": safe_float(resistance),
        "action_zone": zone,
    }


# ----------------------------
# Download bars
# ----------------------------
def download_bars(ticker: str, tf: str, lookback_days: int, threads: bool = True) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    t = ticker.upper().strip()
    tf_l = tf.lower().strip()

    # For 4h/2h: download 60m and resample (more reliable)
    if tf_l in ("4h", "2h"):
        raw = yf.download(
            t,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="60m",
            auto_adjust=True,
            progress=False,
            threads=threads,
        )
        raw = normalize_ohlc(raw)
        if raw.empty:
            return pd.DataFrame()

        rule = "4h" if tf_l == "4h" else "2h"  # lowercase
        return resample_ohlc(raw, rule=rule)

    # Daily / other native
    interval = "1d" if tf_l in ("1d", "d", "day") else tf_l
    raw = yf.download(
        t,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=threads,
    )
    return normalize_ohlc(raw)


# ----------------------------
# Scan
# ----------------------------
def scan(
    tickers: List[str],
    tf: str = "4h",
    lookback_days: int = 180,
    max_bars: int = 350,
    pivot: int = 3,
    sma_fast: int = 20,
    sma_slow: int = 50,
    pullback_min_pct: float = 2.0,
    correction_min_pct: float = 3.0,
    correction_bars: int = 8,
    only_uptrend: bool = False,
    require_setups: bool = False,
    only_correction: bool = False,
    threads: bool = True,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for t in tickers:
        t = t.strip().upper()
        if not t:
            continue

        df = download_bars(t, tf=tf, lookback_days=lookback_days, threads=threads)
        if df.empty:
            continue

        df = df.tail(int(max_bars)).copy()
        if len(df) < max(80, pivot * 2 + max(sma_slow, 50)):
            continue

        piv = compute_pivot_info(df, pivot=pivot)
        trend = trend_filters(df, sma_fast=sma_fast, sma_slow=sma_slow)
        setup = classify_setup(
            df,
            piv,
            trend,
            pullback_min_pct=pullback_min_pct,
            correction_min_pct=correction_min_pct,
            correction_bars=correction_bars,
        )

        if only_uptrend and not setup["uptrend_intact"]:
            continue

        if require_setups and not (setup["setup_hh_hl"] or setup["setup_uptrend_pullback"]):
            continue

        if only_correction and not setup["correction_phase"]:
            continue

        # Rank correction candidates: prefer near support + bigger pullback + controlled slope
        score = 0.0
        score += 1.0 if setup["near_support"] else 0.0
        score += 0.6 if setup["correction_phase"] else 0.0

        pb = setup["pullback_from_high_pct"]
        if pb is not None:
            score += min(pb / 10.0, 0.6)  # cap contribution

        score += 0.3 if trend["sma_fast_gt_slow"] else 0.0
        score += 0.2 if trend["sma_fast_slope_up"] else 0.0

        out.append(
            {
                "ticker": t,
                "tf": tf,
                "bars_used": int(len(df)),
                "pivot": int(pivot),
                **trend,
                "pivot_high_1": safe_float(piv.last_high_1),
                "pivot_high_2": safe_float(piv.last_high_2),
                "pivot_low_1": safe_float(piv.last_low_1),
                "pivot_low_2": safe_float(piv.last_low_2),
                "has_hh": bool(piv.has_hh),
                "has_hl": bool(piv.has_hl),
                "has_hh_hl": bool(piv.has_hh_hl),
                **setup,
                "rank_score": float(score),
            }
        )

    out.sort(key=lambda r: r.get("rank_score", 0.0), reverse=True)

    # One row per ticker
    seen = set()
    deduped = []
    for r in out:
        if r["ticker"] in seen:
            continue
        seen.add(r["ticker"])
        deduped.append(r)

    return deduped


def read_tickers_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s.split()[0].upper())
    return tickers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correction Phase Scanner -> JSON")
    p.add_argument("--tickers", nargs="*", default=[], help="Tickers list")
    p.add_argument("--tickers_file", default="", help="File with one ticker per line")
    p.add_argument("--tf", default="4h", help="Timeframe: 4h (recommended), 2h, 1d, 60m, etc.")
    p.add_argument("--lookback_days", type=int, default=180)
    p.add_argument("--max_bars", type=int, default=350)
    p.add_argument("--pivot", type=int, default=3)
    p.add_argument("--sma_fast", type=int, default=20)
    p.add_argument("--sma_slow", type=int, default=50)

    p.add_argument("--pullback_min_pct", type=float, default=2.0)
    p.add_argument("--correction_min_pct", type=float, default=3.0)
    p.add_argument("--correction_bars", type=int, default=8)

    p.add_argument("--only_uptrend", action="store_true")
    p.add_argument("--require_setups", action="store_true")
    p.add_argument("--only_correction", action="store_true", help="Output only correction-phase tickers")

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

    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "params": {
            "tf": args.tf,
            "lookback_days": args.lookback_days,
            "max_bars": args.max_bars,
            "pivot": args.pivot,
            "sma_fast": args.sma_fast,
            "sma_slow": args.sma_slow,
            "pullback_min_pct": args.pullback_min_pct,
            "correction_min_pct": args.correction_min_pct,
            "correction_bars": args.correction_bars,
            "only_uptrend": bool(args.only_uptrend),
            "require_setups": bool(args.require_setups),
            "only_correction": bool(args.only_correction),
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
        pivot=args.pivot,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        pullback_min_pct=args.pullback_min_pct,
        correction_min_pct=args.correction_min_pct,
        correction_bars=args.correction_bars,
        only_uptrend=bool(args.only_uptrend),
        require_setups=bool(args.require_setups),
        only_correction=bool(args.only_correction),
        threads=not args.no_threads,
    )

    payload["count"] = len(data)
    payload["data"] = data

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(data)} rows to {args.out}")
    if data:
        preview = pd.DataFrame(data)[
            ["ticker", "tf", "close", "correction_phase", "pullback_from_high_pct", "near_support", "action_zone", "rank_score"]
        ].head(20)
        print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
