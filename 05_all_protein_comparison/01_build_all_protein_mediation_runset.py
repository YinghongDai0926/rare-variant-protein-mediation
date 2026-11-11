#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_build_all_protein_mediation_runset.py

Purpose
-------
Build the all-protein mediation runset for:
    encoded protein vs all measured proteins mediation comparison

For each supported gene-disease pair:
    E = rare deleterious coding burden of gene
    Y = incident ICD-3 disease
    M = each measured Olink protein

This script only builds the runset. It does not fit mediation models.

Inputs
------
1. 03_mediation_ranked_significant.tsv.gz (or an equivalent supported-trio master table)
2. wes_olink_covariates_analytic.tsv

Outputs
-------
1. 01_supported_gene_disease_pairs.tsv
2. 01_all_protein_mediation_runset.tsv.gz
3. 01_all_protein_mediation_runset.qc.tsv
"""

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_COVARIATE_COLUMNS = [
    "sample",
    "age",
    "sex",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8",
    "PC9",
    "PC10",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build all-protein mediation runset from supported mediation trios."
    )

    parser.add_argument(
        "--master-table",
        required=True,
        help="Supported mediation trios, e.g. 03_mediation_ranked_significant.tsv.gz; biological annotations are optional.",
    )

    parser.add_argument(
        "--covariate-protein-table",
        required=True,
        help="Path to wes_olink_covariates_analytic.tsv.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--prefix",
        default="01",
        help="Output file prefix. Default: 01.",
    )

    parser.add_argument(
        "--filter-supported",
        action="store_true",
        help=(
            "If set, keep only rows with pass_priority_filter == TRUE. "
            "Use this if master table contains both supported and non-supported trios."
        ),
    )

    parser.add_argument(
        "--strict-encoded-protein-check",
        action="store_true",
        help=(
            "If set, stop when any encoded protein is not found in the Olink protein columns."
        ),
    )

    return parser.parse_args()


def normalize_bool_series(x: pd.Series) -> pd.Series:
    """
    Convert common TRUE/FALSE representations to bool.

    This avoids errors when pass_priority_filter is stored as bool, string, or integer.
    """
    if x.dtype == bool:
        return x

    return (
        x.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "t": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "f": False,
                "0": False,
                "no": False,
                "n": False,
                "nan": False,
                "none": False,
                "": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def read_protein_columns(covariate_protein_table: str) -> list:
    """
    Infer Olink protein columns from the header of wes_olink_covariates_analytic.tsv.

    Scientific logic:
    The downstream all-protein mediation comparison tests each measured Olink protein
    as a potential mediator. Therefore, all columns except sample and covariates are
    treated as mediator protein columns.
    """
    header = pd.read_csv(
        covariate_protein_table,
        sep="\t",
        nrows=0,
    )

    all_columns = list(header.columns)
    covariate_set = set(DEFAULT_COVARIATE_COLUMNS)

    protein_columns = [col for col in all_columns if col not in covariate_set]

    if len(protein_columns) == 0:
        raise ValueError(
            "No protein columns were detected. "
            "Please check wes_olink_covariates_analytic.tsv header."
        )

    return protein_columns


def prepare_supported_pairs(master_table: str, filter_supported: bool) -> pd.DataFrame:
    """
    Prepare unique supported gene-disease pairs.

    The encoded protein is taken from the original supported mediation trio.
    In the current framework, protein column name and gene symbol are expected to match.
    """
    df = pd.read_csv(master_table, sep="\t")

    required_cols = [
        "trio_id",
        "gene",
        "protein",
        "icd3",
        "tier_label",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns in master table: "
            + ", ".join(missing_cols)
        )

    if filter_supported:
        if "pass_priority_filter" not in df.columns:
            raise ValueError(
                "--filter-supported was set, but pass_priority_filter column is missing."
            )
        df = df.loc[normalize_bool_series(df["pass_priority_filter"])].copy()

    rename_map = {
        "trio_id": "original_trio_id",
        "protein": "encoded_protein",
    }

    keep_cols = [
        "trio_id",
        "gene",
        "protein",
        "icd3",
        "tier_label",
    ]

    optional_cols = [
        "indirect_logHR",
        "indirect_hr",
        "proportion_mediated_logscale",
        "indirect_p_boot",
        "indirect_q_boot",
        "pass_model_success",
        "pass_boot_stability",
        "pass_indirect_significance",
        "pass_priority_filter",
        "disease_system",
    ]

    keep_cols += [col for col in optional_cols if col in df.columns]

    pairs = (
        df.loc[:, keep_cols]
        .rename(columns=rename_map)
        .drop_duplicates(subset=["gene", "icd3", "encoded_protein"])
        .copy()
    )

    pairs["pair_id"] = pairs["gene"].astype(str) + "__" + pairs["icd3"].astype(str)

    # Put identifiers first.
    first_cols = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "tier_label",
        "original_trio_id",
    ]
    other_cols = [col for col in pairs.columns if col not in first_cols]
    pairs = pairs.loc[:, first_cols + other_cols]

    return pairs


def build_runset(pairs: pd.DataFrame, protein_columns: list) -> pd.DataFrame:
    """
    Expand each supported gene-disease pair to all measured Olink proteins.

    Scientific logic:
    For the same rare coding burden-disease pair, every measured circulating protein
    is tested as a mediator, so that the encoded protein can be ranked against the
    full measured protein panel.
    """
    protein_df = pd.DataFrame({"mediator_protein": protein_columns})

    pairs_for_cross = pairs[
        [
            "pair_id",
            "gene",
            "icd3",
            "encoded_protein",
            "tier_label",
            "original_trio_id",
        ]
    ].copy()

    pairs_for_cross["_key"] = 1
    protein_df["_key"] = 1

    runset = (
        pairs_for_cross
        .merge(protein_df, on="_key", how="inner")
        .drop(columns=["_key"])
    )

    runset["is_encoded_protein"] = (
        runset["mediator_protein"].astype(str)
        == runset["encoded_protein"].astype(str)
    ).astype(int)

    return runset


def write_qc(
    qc_path: Path,
    pairs: pd.DataFrame,
    runset: pd.DataFrame,
    protein_columns: list,
):
    encoded_available = pairs["encoded_protein"].isin(protein_columns)

    qc_items = [
        ("n_supported_gene_disease_rows", len(pairs)),
        ("n_unique_pair_id", pairs["pair_id"].nunique()),
        ("n_unique_genes", pairs["gene"].nunique()),
        ("n_unique_icd3", pairs["icd3"].nunique()),
        ("n_olink_proteins", len(protein_columns)),
        ("n_all_protein_tests", len(runset)),
        ("n_encoded_protein_found", int(encoded_available.sum())),
        ("n_encoded_protein_missing", int((~encoded_available).sum())),
        (
            "encoded_protein_missing_list",
            ",".join(
                sorted(
                    pairs.loc[~encoded_available, "encoded_protein"]
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),
        ),
        (
            "n_encoded_rows_in_runset",
            int(runset["is_encoded_protein"].sum()),
        ),
    ]

    qc = pd.DataFrame(qc_items, columns=["metric", "value"])
    qc.to_csv(qc_path, sep="\t", index=False)


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    supported_pairs_path = outdir / f"{args.prefix}_supported_gene_disease_pairs.tsv"
    runset_path = outdir / f"{args.prefix}_all_protein_mediation_runset.tsv.gz"
    qc_path = outdir / f"{args.prefix}_all_protein_mediation_runset.qc.tsv"

    protein_columns = read_protein_columns(args.covariate_protein_table)

    pairs = prepare_supported_pairs(
        master_table=args.master_table,
        filter_supported=args.filter_supported,
    )

    encoded_missing = sorted(
        set(pairs["encoded_protein"].astype(str)) - set(protein_columns)
    )

    if args.strict_encoded_protein_check and encoded_missing:
        raise ValueError(
            "Some encoded proteins were not found in Olink protein columns: "
            + ", ".join(encoded_missing)
        )

    runset = build_runset(pairs=pairs, protein_columns=protein_columns)

    pairs.to_csv(supported_pairs_path, sep="\t", index=False)
    runset.to_csv(runset_path, sep="\t", index=False, compression="gzip")
    write_qc(
        qc_path=qc_path,
        pairs=pairs,
        runset=runset,
        protein_columns=protein_columns,
    )

    print("[Done] All-protein mediation runset generated.")
    print(f"[Output] {supported_pairs_path}")
    print(f"[Output] {runset_path}")
    print(f"[Output] {qc_path}")
    print(f"[QC] n_supported_gene_disease_rows = {len(pairs)}")
    print(f"[QC] n_olink_proteins = {len(protein_columns)}")
    print(f"[QC] n_all_protein_tests = {len(runset)}")
    print(f"[QC] n_encoded_rows_in_runset = {int(runset['is_encoded_protein'].sum())}")


if __name__ == "__main__":
    main()
