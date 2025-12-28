#!/usr/bin/env python3
"""
HH/HL (Higher High + Higher Low) Scanner - DAILY timeframe (LAST 1 MONTH)

This version is designed to align with TradingView "1M" view by restricting
analysis to the last ~1 month of daily candles (last N calendar days, then tail ~25 bars).

- Reads tickers from a text file (one per line)
- Downloads DAILY OHLC via yfinance
- Restricts to last N calendar days (default 35) and then keeps last ~25 trading days
- Detects pivot (swing) highs/lows using a left/right window
- Declares HH/HL if the most recent 2 pivot highs are increasing (HH)
  AND the most recent 2 pivot lows are increasing (HL)
- Outputs JSON to --out (folders auto-created)

Example:
  python hh_hl_scanner_1m.py --tickers_file data/nifty50.txt --last_days 35 --pivot 2 --out api/daily_patterns/hhhl_nifty_1m.json
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


# ----------------------------
# Utilities
# ----------------------------
def read_tickers_file(path: str) -> List[str]:
    tickers: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            tickers.append(s.split()[0].upper())
    return tickers


def find_pivots(df: pd.DataFrame, left_right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Pivot High at index i if High[i] is strictly greater than highs in [i-left_right .. i+left_right]
    Pivot Low  at index i if Low[i]  is strictly lower  than lows  in [i-left_right .. i+left_right]
    Returns boolean series: pivot_high, pivot_low
    """
    h = df["High"]
    l = df["Low"]
    n = int(left_right)

    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)

    # Only evaluate pivots where both sides exist
    for i in range(n, len(df) - n):
        window_h = h.iloc[i - n : i + n + 1]
        window_l = l.iloc[i - n : i + n + 1]

        # Strict max/min avoids ties noise
        if h.iloc[i] == window_h.max() and (window_h == h.iloc[i]).sum() == 1:
            pivot_high.iloc[i] = True

        if l.iloc[i] == window_l.min() and (window_l == l.iloc[i]).sum() == 1:
            pivot_low.iloc[i] = True

    return pivot_high, pivot_low


def hh_hl_from_pivots(
    df: pd.DataFrame,
    pivot_high: pd.Series,
    pivot_low: pd.Series,
    min_pivots: int = 2,
) -> Dict[str, Any]:
    """
    Determine if we have HH/HL:
      last 2 pivot highs: H2 > H1  (higher high)
      last 2 pivot lows : L2 > L1  (higher low)
    Returns details + booleans.
    """
    ph_idx = df.index[pivot_high]
    pl_idx = df.index[pivot_low]

    result: Dict[str, Any] = {
        "has_hh": False,
        "has_hl": False,
        "has_hh_hl": False,
        "last_high_1_date": None,
        "last_high_2_date": None,
        "last_low_1_date": None,
        "last_low_2_date": None,
        "last_high_1": None,
        "last_high_2": None,
        "last_low_1": None,
        "last_low_2": None,
        "pivot_high_count": int(len(ph_idx)),
        "pivot_low_count": int(len(pl_idx)),
    }

    if len(ph_idx) < min_pivots or len(pl_idx) < min_pivots:
        return result

    # last two pivot highs (older -> newer)
    h1_date, h2_date = ph_idx[-2], ph_idx[-1]
    h1, h2 = float(df.loc[h1_date, "High"]), float(df.loc[h2_date, "High"])
    has_hh = h2 > h1

    # last two pivot lows (older -> newer)
    l1_date, l2_date = pl_idx[-2], pl_idx[-1]
    l1, l2 = float(df.loc[l1_date, "Low"]), float(df.loc[l2_date, "Low"])
    has_hl = l2 > l1

    result.update(
        {
            "has_hh": bool(has_hh),
            "has_hl": bool(has_hl),
            "has_hh_hl": bool(has_hh and has_hl),
            "last_high_1_date": str(pd.to_datetime(h1_date).date()),
            "last_high_2_date": str(pd.to_datetime(h2_date).date()),
            "last_low_1_date": str(pd.to_datetime(l1_date).date()),
            "last_low_2_date": str(pd.to_datetime(l2_date).date()),
            "last_high_1": h1,
            "last_high_2": h2,
            "last_low_1": l1,
            "last_low_2": l2,
        }
    )
    return result


# ----------------------------
# Scanner
# ----------------------------
def scan_hh_hl(
    tickers: List[str],
    last_days: int = 35,          # ~1 month calendar days + weekends/holidays buffer
    max_trading_days: int = 25,   # last ~1 trading month
    pivot: int = 2,
    min_pivots: int = 2,
    threads: bool = True,
) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(last_days))

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
            df = data[t].dropna()
            if df.empty:
                continue

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

            # enforce last ~1 trading month
            df = df.tail(int(max_trading_days))

            # need enough bars to form pivots
            if len(df) < (pivot * 2 + 6):
                continue

            ph, pl = find_pivots(df, left_right=pivot)
            info = hh_hl_from_pivots(df, ph, pl, min_pivots=min_pivots)

            last_close = float(df["Close"].iloc[-1])
            last_date = str(pd.to_datetime(df.index[-1]).date())

            rows.append(
                {
                    "ticker": t,
                    "last_date": last_date,
                    "last_close": last_close,
                    "lookback_calendar_days": int(last_days),
                    "lookback_trading_days": int(len(df)),
                    "pivot_window": int(pivot),
                    **info,
                }
            )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sort: HHHL first, then most recent date
    out["hhhl_rank"] = out["has_hh_hl"].astype(int)
    out = out.sort_values(["hhhl_rank", "last_date"], ascending=[False, False]).reset_index(drop=True)
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan tickers for HH/HL pattern on DAILY timeframe (last ~1 month).")
    p.add_argument("--tickers_file", default="nifty.txt", help="Text file with one ticker per line.")
    p.add_argument("--last_days", type=int, default=35, help="Analyze last N calendar days (default 35).")
    p.add_argument("--max_trading_days", type=int, default=25, help="Keep last N trading days from downloaded data (default 25).")
    p.add_argument("--pivot", type=int, default=2, help="Pivot window size (left/right). 2 is best for 1M view.")
    p.add_argument("--min_pivots", type=int, default=2, help="Minimum pivot highs/lows required.")
    p.add_argument("--out", default="hh_hl_output.json", help="Output JSON path.")
    p.add_argument("--no_threads", action="store_true", help="Disable yfinance threading.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    tickers = read_tickers_file(args.tickers_file)

    df = scan_hh_hl(
        tickers=tickers,
        last_days=args.last_days,
        max_trading_days=args.max_trading_days,
        pivot=args.pivot,
        min_pivots=args.min_pivots,
        threads=not args.no_threads,
    )

    # Convert pandas NaN to Python None so JSON becomes strict (null)
    if not df.empty:
        df = df.replace({np.nan: None})


    payload: Dict[str, Any] = {
        "updated_utc": pd.Timestamp.utcnow().isoformat(),
        "params": {
            "tickers_file": args.tickers_file,
            "last_days": args.last_days,
            "max_trading_days": args.max_trading_days,
            "pivot": args.pivot,
            "min_pivots": args.min_pivots,
        },
        "count": int(len(df)),
        "count_hhhl": int(df["has_hh_hl"].sum()) if not df.empty else 0,
        "data": df.to_dict(orient="records") if not df.empty else [],
    }

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Wrote {payload['count']} rows to: {args.out}")

    if not df.empty:
        show_cols = [
            "ticker",
            "has_hh_hl",
            "has_hh",
            "has_hl",
            "pivot_high_count",
            "pivot_low_count",
            "last_high_1_date",
            "last_high_2_date",
            "last_low_1_date",
            "last_low_2_date",
            "last_close",
        ]
        print(df[show_cols].head(30).to_string(index=False))

        picks = df[df["has_hh_hl"] == True]["ticker"].tolist()
        print("\nHH+HL tickers (last ~1 month, daily):")
        print(", ".join(picks) if picks else "(none found)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
