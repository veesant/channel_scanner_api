#!/usr/bin/env python3
import pandas as pd
import requests

URL = "https://en.wikipedia.org/wiki/NIFTY_50"

def main():
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # Fetch HTML with a browser-like header (avoids 403)
    resp = requests.get(URL, headers=headers, timeout=30)
    resp.raise_for_status()

    # Parse tables from the HTML string (not from the URL)
    tables = pd.read_html(resp.text)

    # Find the constituents table robustly (don’t rely on a fixed index)
    constituents = None
    for t in tables:
        cols = {c.strip().lower() for c in t.columns.astype(str)}
        if "symbol" in cols and ("company name" in cols or "company" in cols):
            constituents = t
            break

    # Fallback: Wikipedia layout sometimes changes; try a simpler match
    if constituents is None:
        for t in tables:
            cols = {c.strip().lower() for c in t.columns.astype(str)}
            if "symbol" in cols:
                constituents = t
                break

    if constituents is None:
        raise RuntimeError("Could not find a table with a 'Symbol' column on the page.")

    # Extract symbols and convert to Yahoo Finance NSE format
    symbol_col = [c for c in constituents.columns if str(c).strip().lower() == "symbol"][0]
    symbols = constituents[symbol_col].astype(str).str.strip().tolist()

    # Append .NS for yfinance
    symbols = [s + ".NS" for s in symbols if s and s != "nan"]

    out_file = "nifty50.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for s in symbols:
            f.write(s + "\n")

    print(f"Saved {len(symbols)} tickers to {out_file}")
    print("Sample:", symbols[:10])

if __name__ == "__main__":
    main()
