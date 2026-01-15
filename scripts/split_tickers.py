from pathlib import Path

def split_file(src_path, prefix, chunk):
    src = Path(src_path)
    lines = [
        l.strip() for l in src.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]

    out_dir = src.parent
    for i in range(0, len(lines), chunk):
        idx = i // chunk + 1
        out = out_dir / f"{prefix}_{idx:02d}.txt"
        out.write_text("\n".join(lines[i:i+chunk]) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({min(chunk, len(lines)-i)} tickers)")

if __name__ == "__main__":
    split_file("data/nasdaq.txt", "nasdaq", 520)
    split_file("data/nyse.txt", "nyse", 520)
