#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
from typing import Dict, List, Set, Tuple

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


def read_list_file(path: str) -> List[str]:
    vals = []
    with open(path, "r") as f:
        for line in f:
            x = line.strip()
            if x != "":
                vals.append(x)
    return vals


def parse_gmt(gmt_path: str) -> List[Tuple[str, str, Set[str]]]:
    """
    Parse GMT file:
      col1 = term_name
      col2 = description/url
      col3+ = gene/protein symbols
    """
    terms = []
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term_name = parts[0].strip()
            term_desc = parts[1].strip()
            members = set([x.strip() for x in parts[2:] if x.strip() != ""])
            if term_name != "" and len(members) > 0:
                terms.append((term_name, term_desc, members))
    return terms


def run_ora(
    input_set: Set[str],
    background_set: Set[str],
    term_records: List[Tuple[str, str, Set[str]]],
    min_term_size: int,
    max_term_size: int,
) -> pd.DataFrame:
    """
    ORA using Fisher exact test.
    Universe is the union of all members of the supplied GMT (before term-size filtering).
    """
    input_in_bg = input_set & background_set
    n_input = len(input_in_bg)
    n_bg = len(background_set)

    rows = []

    for term_name, term_desc, members in term_records:
        members_bg = members & background_set
        term_size = len(members_bg)

        if term_size < min_term_size or term_size > max_term_size:
            continue

        overlap = input_in_bg & members_bg
        n_overlap = len(overlap)

        if n_overlap == 0:
            continue

        # 2x2 table
        a = n_overlap
        b = n_input - a
        c = term_size - a
        d = n_bg - a - b - c

        if min(a, b, c, d) < 0:
            continue

        # enrichment test: greater
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")

        expected = (n_input * term_size) / n_bg if n_bg > 0 else math.nan
        fold_enrichment = (a / expected) if expected and expected > 0 else math.nan

        rows.append({
            "term_name": term_name,
            "term_description": term_desc,
            "term_size_in_background": term_size,
            "n_input_proteins": n_input,
            "n_overlap": a,
            "overlap_proteins": ";".join(sorted(overlap)),
            "expected_overlap": expected,
            "fold_enrichment": fold_enrichment,
            "p": p,
            "a_overlap": a,
            "b_input_not_in_term": b,
            "c_term_not_in_input": c,
            "d_neither": d,
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "term_name", "term_description", "term_size_in_background",
            "n_input_proteins", "n_overlap", "overlap_proteins",
            "expected_overlap", "fold_enrichment", "p",
            "a_overlap", "b_input_not_in_term", "c_term_not_in_input", "d_neither",
            "fdr_q"
        ])

    out = pd.DataFrame(rows)
    out["fdr_q"] = multipletests(out["p"].values, method="fdr_bh")[1]
    out = out.sort_values(["fdr_q", "p", "n_overlap", "fold_enrichment"], ascending=[True, True, False, False]).copy()
    return out



def annotate_ontology(results):
    """Label GO namespaces AFTER the original joint BH calculation.

    Splits retain the original fdr_q values. Unknown term prefixes are kept
    and marked Unclassified; Reactome terms are kept and labelled Reactome.
    """
    result = results.copy()
    def classify(name):
        name = str(name).upper()
        for prefix, ontology in (("GOBP_", "BP"), ("GOMF_", "MF"), ("GOCC_", "CC")):
            if name.startswith(prefix):
                return ontology
        if name.startswith("REACTOME_"):
            return "Reactome"
        return "Unclassified"
    result["ontology"] = result["term_name"].map(classify)
    result["background_method"] = "gmt_universe"
    result["fdr_scope"] = "joint_tested_terms_in_supplied_gmt"
    return result


def main():
    parser = argparse.ArgumentParser(description="Run ORA enrichment for a protein set against a GMT database (Universal Background)")
    parser.add_argument("--input_list", required=True, help="Protein list txt, one protein per line")
    parser.add_argument("--gmt", required=True, help="GMT file")
    parser.add_argument("--set_name", required=True, help="Label, e.g. Tier1 or Tier2")
    parser.add_argument("--database_label", required=True, help="Label, e.g. Reactome or GO (BP, MF and CC jointly)")
    parser.add_argument("--out_tsv", required=True, help="Output enrichment table")
    parser.add_argument("--out_qc", required=True, help="Output QC table")
    parser.add_argument("--min_term_size", type=int, default=5)
    parser.add_argument("--max_term_size", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_tsv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_qc)), exist_ok=True)

    input_list = sorted(set(read_list_file(args.input_list)))
    
    # 1. Parse GMT file first
    gmt_terms = parse_gmt(args.gmt)

    # 2. Dynamically construct the universal background set from the GMT database
    background_set = set()
    for _, _, members in gmt_terms:
        background_set.update(members)
        
    background_list = sorted(background_set)
    input_set = set(input_list)

    # 3. Intersect inputs with the universal background
    input_in_bg = input_set & background_set
    input_not_in_bg = input_set - background_set

    # 4. Run ORA
    res = run_ora(
        input_set=input_in_bg,
        background_set=background_set,
        term_records=gmt_terms,
        min_term_size=args.min_term_size,
        max_term_size=args.max_term_size
    )

    if not res.empty:
        res.insert(0, "database", args.database_label)
        res.insert(0, "set_name", args.set_name)
    else:
        res = pd.DataFrame(columns=[
            "set_name", "database", "term_name", "term_description",
            "term_size_in_background", "n_input_proteins", "n_overlap",
            "overlap_proteins", "expected_overlap", "fold_enrichment",
            "p", "a_overlap", "b_input_not_in_term", "c_term_not_in_input",
            "d_neither", "fdr_q"
        ])

    res = annotate_ontology(res)
    res.to_csv(args.out_tsv, sep="\t", index=False)
    # No ontology-wise FDR recomputation: these are views of the same joint result.
    if args.database_label.upper().startswith("GO") or res["ontology"].isin(["BP", "MF", "CC"]).any():
        from pathlib import Path
        path = Path(args.out_tsv)
        for ontology in ("BP", "MF", "CC"):
            split_path = path.with_name(path.stem + "." + ontology + path.suffix)
            res.loc[res["ontology"].eq(ontology)].to_csv(split_path, sep="\t", index=False)

    # Output QC (Maintains the exact same format, but n_background now reflects the GMT universe size)
    qc_rows = [
        {"metric": "set_name", "value": args.set_name},
        {"metric": "database_label", "value": args.database_label},
        {"metric": "n_input_raw", "value": len(input_list)},
        {"metric": "n_background_from_gmt", "value": len(background_list)},
        {"metric": "n_input_in_background", "value": len(input_in_bg)},
        {"metric": "n_input_not_in_background", "value": len(input_not_in_bg)},
        {"metric": "n_gmt_terms_total", "value": len(gmt_terms)},
        {"metric": "n_enriched_terms_output", "value": res.shape[0]},
        {"metric": "n_fdr_lt_0_05", "value": int((pd.to_numeric(res["fdr_q"], errors="coerce") < 0.05).sum()) if not res.empty else 0},
        {"metric": "n_nominal_p_lt_0_05", "value": int((pd.to_numeric(res["p"], errors="coerce") < 0.05).sum()) if not res.empty else 0},
        {"metric": "min_term_size", "value": args.min_term_size},
        {"metric": "max_term_size", "value": args.max_term_size},
    ]

    if len(input_not_in_bg) > 0:
        qc_rows.append({"metric": "input_not_in_background_members", "value": ";".join(sorted(input_not_in_bg))})

    if not res.empty:
        qc_rows.extend([
            {"metric": "top_term_name", "value": str(res.iloc[0]["term_name"])},
            {"metric": "top_term_fdr_q", "value": float(res.iloc[0]["fdr_q"])},
            {"metric": "top_term_n_overlap", "value": int(res.iloc[0]["n_overlap"])},
        ])

    qc_rows.extend([
        {"metric": "background_method", "value": "gmt_universe"},
        {"metric": "fdr_scope", "value": "joint_tested_terms_in_supplied_gmt"},
    ])
    for ontology in ("BP", "MF", "CC", "Reactome", "Unclassified"):
        qc_rows.append({"metric": "n_ontology_" + ontology, "value": int(res["ontology"].eq(ontology).sum())})
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(args.out_qc, sep="\t", index=False)

    print(f"[OK] Enrichment results written to: {args.out_tsv}")
    print(f"[OK] QC written to: {args.out_qc}")
    print(f"[INFO] set={args.set_name}, db={args.database_label}, terms={res.shape[0]}, background_size={len(background_list)}")


if __name__ == "__main__":
    main()