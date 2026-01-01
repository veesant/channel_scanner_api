#!/usr/bin/env python3
"""
Early Double Bottom Scanner (Daily) - "Beginning of second bottom bounce"

Goal:
  Find tickers where:
    - A double-bottom structure exists (two similar pivot lows L1, L2)
    - L2 is recent (just printed)
    - Price has started turning up from L2 (early uptrend starting)
    - Still below neckline (pre-breakout)
    - Optionally near neckline (actionable zone)

Outputs strict JSON (no NaN) to --out.

Example:
  python double_bottom_early_scanner.py \
    --tickers_file data/nifty50.txt --tickers_file data/nasdaq100.txt --tickers_file data/russel.txt \
    --lookback_days 200 --pivot 3 \
    --bottom_tol_pct 2.5 --min_sep_days 10 --min_bounce_pct 6 \
    --l2_max_age_days 7 --turn_up_pct 1.0 \
    --sma_fast 5 --sma_slow 10 \
    --prebreak_margin_pct 0.5 --near_neckline_pct 8 \
    --out api/double_bottom/early_double_bottom_all.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def read_tickers_file(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0].upper())
    return out


def find_pivots(df: pd.DataFrame, left_right: int = 3) -> Tuple[pd.Series, pd.Series]:
    h = df["High"]
    l = df["Low"]
    n = int(left_right)
    ph = pd.Series(False, index=df.index)
    pl = pd.Series(False, index=df.index)

    for i in range(n, len(df) - n):
        wh = h.iloc[i - n : i + n + 1]
        wl = l.iloc[i - n : i + n + 1]
        if h.iloc[i] == wh.max() and (wh == h.iloc[i]).sum() == 1:
            ph.iloc[i] = True
        if l.iloc[i] == wl.min() and (wl == l.iloc[i]).sum() == 1:
            pl.iloc[i] = True

    return ph, pl


def extract_ticker_df(data: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    if data is None:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        cols = []
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if (field, ticker) in data.columns:
                cols.append((field, ticker))
        if len(cols) >= 4:
            df = data[cols].copy()
            df.columns = [c[0] for c in df.columns]
            return df

        cols = []
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            if (ticker, field) in data.columns:
                cols.append((ticker, field))
        if len(cols) >= 4:
            df = data[cols].copy()
            df.columns = [c[1] for c in df.columns]
            return df

        return None

    # single ticker DF
    if set(["Open", "High", "Low", "Close"]).issubset(set(map(str, data.columns))):
        return data.copy()

    try:
        if ticker in data:
            return data[ticker].copy()
    except Exception:
        pass

    return None


def add_sma(df: pd.DataFrame, n: int, col: str = "Close") -> pd.Series:
    return df[col].rolling(n).mean()


def scan_early_double_bottom_for_df(
    df: pd.DataFrame,
    pivot: int,
    bottom_tol_pct: float,
    min_sep_days: int,
    min_bounce_pct: float,
    l2_max_age_days: int,
    turn_up_pct: float,
    sma_fast: int,
    sma_slow: int,
    prebreak_margin_pct: float,
    near_neckline_pct: float,
) -> Optional[Dict[str, Any]]:
    df = df.dropna().copy()
    if df.empty or len(df) < (pivot * 2 + 40):
        return None

    # SMAs for "starting uptrend"
    df["sma_fast"] = add_sma(df, sma_fast)
    df["sma_slow"] = add_sma(df, sma_slow)

    ph, pl = find_pivots(df, left_right=pivot)
    low_idx = df.index[pl]
    high_idx = df.index[ph]

    if len(low_idx) < 2:
        return None

    # Last two pivot lows
    l1_date, l2_date = low_idx[-2], low_idx[-1]
    l1_dt = pd.to_datetime(l1_date)
    l2_dt = pd.to_datetime(l2_date)

    # Need separation
    if (l2_dt - l1_dt).days < min_sep_days:
        return None

    l1 = float(df.loc[l1_date, "Low"])
    l2 = float(df.loc[l2_date, "Low"])
    if l1 <= 0:
        return None

    # Similar bottoms
    tol = float(bottom_tol_pct) / 100.0
    if abs(l2 - l1) / l1 > tol:
        return None

    # Neckline (highest pivot high between the bottoms)
    between_highs = [d for d in high_idx if l1_date < d < l2_date]
    if not between_highs:
        return None
    neckline_date = max(between_highs, key=lambda d: float(df.loc[d, "High"]))
    neckline = float(df.loc[neckline_date, "High"])

    bottom_avg = (l1 + l2) / 2.0
    bounce_pct = (neckline / bottom_avg - 1.0) * 100.0
    if bounce_pct < float(min_bounce_pct):
        return None

    # Recentness: L2 should be very recent (early stage)
    last_idx = df.index[-1]
    last_dt = pd.to_datetime(last_idx)

    # Use trading-day age approximation: count rows since L2
    try:
        l2_pos = df.index.get_loc(l2_date)
    except Exception:
        return None
    bars_since_l2 = (len(df) - 1) - int(l2_pos)
    if bars_since_l2 < 0:
        return None
    if bars_since_l2 > int(l2_max_age_days):
        return None

    last_close = float(df["Close"].iloc[-1])

    # Started turning up from L2
    up_from_l2_pct = (last_close / l2 - 1.0) * 100.0
    if up_from_l2_pct < float(turn_up_pct):
        return None

    # SMA momentum (fast above slow) to signal early uptrend
    sma_f = df["sma_fast"].iloc[-1]
    sma_s = df["sma_slow"].iloc[-1]
    if not (np.isfinite(sma_f) and np.isfinite(sma_s)):
        return None
    if float(sma_f) <= float(sma_s):
        return None

    # Must still be below neckline (pre-breakout)
    prebreak_margin = float(prebreak_margin_pct) / 100.0
    if last_close >= neckline * (1.0 - prebreak_margin):
        return None

    # Should not be too far from neckline (actionable)
    near_band = float(near_neckline_pct) / 100.0
    dist_to_neckline_pct = abs(neckline - last_close) / neckline * 100.0
    if dist_to_neckline_pct > float(near_neckline_pct):
        return None

    # Score: prefer closer bottoms + bigger bounce + closer to neckline (but still below)
    similarity_pct = (1.0 - abs(l2 - l1) / l1) * 100.0
    closeness_bonus = max(0.0, (near_neckline_pct - dist_to_neckline_pct))  # higher is better
    quality = (min(100.0, similarity_pct) * 0.45) + (min(40.0, bounce_pct) * 1.25) + (closeness_bonus * 0.8)

    return {
        "l1_date": str(l1_dt.date()),
        "l2_date": str(l2_dt.date()),
        "l1_low": l1,
        "l2_low": l2,
        "neckline_date": str(pd.to_datetime(neckline_date).date()),
        "neckline": neckline,
        "bounce_pct": float(bounce_pct),
        "bottom_similarity_pct": float(similarity_pct),
        "bars_since_l2": int(bars_since_l2),
        "up_from_l2_pct": float(up_from_l2_pct),
        "sma_fast": int(sma_fast),
        "sma_slow": int(sma_slow),
        "last_date": str(last_dt.date()),
        "last_close": float(last_close),
        "dist_to_neckline_pct": float(dist_to_neckline_pct),
        "quality_score": float(quality),
    }


def scan_early_double_bottom(
    tickers: List[str],
    lookback_days: int,
    pivot: int,
    bottom_tol_pct: float,
    min_sep_days: int,
    min_bounce_pct: float,
    l2_max_age_days: int,
    turn_up_pct: float,
    sma_fast: int,
    sma_slow: int,
    prebreak_margin_pct: float,
    near_neckline_pct: float,
    threads: bool = True,
) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))

    data = yf.download(
        tickers=tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="ticker",
        threads=threads,
        progress=False,
        interval="1d",
    )

    rows: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            df = extract_ticker_df(data, t)
            if df is None or df.empty:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            info = scan_early_double_bottom_for_df(
                df=df,
                pivot=pivot,
                bottom_tol_pct=bottom_tol_pct,
                min_sep_days=min_sep_days,
                min_bounce_pct=min_bounce_pct,
                l2_max_age_days=l2_max_age_days,
                turn_up_pct=turn_up_pct,
                sma_fast=sma_fast,
                sma_slow=sma_slow,
                prebreak_margin_pct=prebreak_margin_pct,
                near_neckline_pct=near_neckline_pct,
            )
            if not info:
                continue

            rows.append({"ticker": t, **info})
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["quality_score", "bounce_pct"], ascending=[False, False]).reset_index(drop=True)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Early Double Bottom scanner (Daily): second bottom just formed, turning up.")
    p.add_argument("--tickers_file", action="append", default=[], help="Ticker file path (repeatable).")

    p.add_argument("--lookback_days", type=int, default=200, help="Calendar-day lookback to fetch daily data.")
    p.add_argument("--pivot", type=int, default=3, help="Pivot window size (left/right).")

    p.add_argument("--bottom_tol_pct", type=float, default=2.5, help="Bottom similarity tolerance percent.")
    p.add_argument("--min_sep_days", type=int, default=10, help="Min separation (days) between bottoms.")
    p.add_argument("--min_bounce_pct", type=float, default=6.0, help="Min bounce percent from bottoms to neckline.")

    p.add_argument("--l2_max_age_days", type=int, default=7, help="Second bottom must be within last N trading bars.")
    p.add_argument("--turn_up_pct", type=float, default=1.0, help="Close must be at least this % above L2 low.")

    p.add_argument("--sma_fast", type=int, default=5, help="Fast SMA length for early uptrend.")
    p.add_argument("--sma_slow", type=int, default=10, help="Slow SMA length for early uptrend.")

    p.add_argument("--prebreak_margin_pct", type=float, default=0.5, help="Must be at least this % below neckline.")
    p.add_argument("--near_neckline_pct", type=float, default=8.0, help="Must be within this % of neckline to be actionable.")

    p.add_argument("--out", default="api/double_bottom/early_double_bottom_all.json", help="Output JSON path.")
    p.add_argument("--no_threads", action="store_true", help="Disable yfinance threading.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    tickers: List[str] = []
    for fp in args.tickers_file:
        tickers.extend(read_tickers_file(fp))
    tickers = sorted(set(tickers))

    df = scan_early_double_bottom(
        tickers=tickers,
        lookback_days=args.lookback_days,
        pivot=args.pivot,
        bottom_tol_pct=args.bottom_tol_pct,
        min_sep_days=args.min_sep_days,
        min_bounce_pct=args.min_bounce_pct,
        l2_max_age_days=args.l2_max_age_days,
        turn_up_pct=args.turn_up_pct,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        prebreak_margin_pct=args.prebreak_margin_pct,
        near_neckline_pct=args.near_neckline_pct,
        threads=not args.no_threads,
    )

    # strict JSON
    if not df.empty:
        df = df.replace({np.nan: None})

    payload: Dict[str, Any] = {
        "updated_utc": pd.Timestamp.utcnow().isoformat(),
        "params": vars(args),
        "count": int(len(df)),
        "data": df.to_dict(orient="records") if not df.empty else [],
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Wrote {payload['count']} rows to: {args.out}")
    if not df.empty:
        print(df[["ticker", "quality_score", "bars_since_l2", "up_from_l2_pct", "bounce_pct", "dist_to_neckline_pct"]]
              .head(25).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
