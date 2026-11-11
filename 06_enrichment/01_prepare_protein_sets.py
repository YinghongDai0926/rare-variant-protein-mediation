#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

def read_symbols(path, protein_col):
    df = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    if protein_col not in df:
        raise ValueError(f"{path} is missing {protein_col}")
    symbols = df[protein_col].dropna().astype(str).str.strip()
    return sorted(set(symbols[symbols.ne("")]))

def main():
    parser = argparse.ArgumentParser(description="Prepare existing Tier1/Tier2 protein sets for GMT-universe ORA")
    parser.add_argument("--tier1_tsv", required=True)
    parser.add_argument("--tier2_tsv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--protein_col", default="protein")
    args = parser.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t1 = read_symbols(args.tier1_tsv, args.protein_col)
    t2 = read_symbols(args.tier2_tsv, args.protein_col)
    sets = {"tier1": t1, "tier2": t2, "tier1_2_combined": sorted(set(t1) | set(t2))}
    qc = []
    for label, symbols in sets.items():
        (out / f"{label}_protein_list.txt").write_text("".join(f"{x}\n" for x in symbols))
        pd.DataFrame({"protein": symbols, "set_name": label}).to_csv(out / f"{label}_protein_table.tsv", sep="\t", index=False)
        qc.append({"metric": f"n_{label}_proteins", "value": len(symbols)})
    pd.DataFrame(qc).to_csv(out / "01_prepare_protein_sets.qc.tsv", sep="\t", index=False)
    print(f"[OK] Protein sets written to {out}; GMT defines the universe during ORA.")

if __name__ == "__main__":
    main()
