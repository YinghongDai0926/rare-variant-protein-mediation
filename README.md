# Rare-variant protein mediation: core analyses

This release contains 27 existing core Python analysis programs and 27 corresponding portable Bash launchers. It covers exposure–mediator, mediator–outcome, exposure–outcome, primary mediation, all-protein comparison and GMT-universe enrichment. Sensitivity analyses, standalone plotting, case-card interpretation and missing upstream generators are outside this release.

The original source scripts are preserved separately. `docs/source_mapping.tsv` records the original path, new path and source hash for every retained analysis program and launcher.

## Environment

The author-reported analysis versions are **Python 3.12.6** and **R 4.5.1 (2025-06-13), “Great Square Root”**. The retained core programs are Python; R is recorded for the study environment and is not invoked by this subset.

```bash
conda env create -f environment.yml
conda activate protein-mediation-core
cp config.example.sh config.sh
```

Set `PYTHON_BIN` and the external input locations in `config.sh`. Historical versions of individual Python/R packages were not provided; no package-version lock has been invented. `requirements.txt` lists the Python dependencies. See `docs/validation.md` for the actual runtime used for this release's validation, which differs from the author-reported environment.

Launchers load `config.sh`, falling back to `config.example.sh`. `CONFIG_FILE` can select another configuration. Paths containing spaces are supported. Relative default input paths are examples of the input contract, not bundled study data.

## Inputs and scope boundary

Supply the preprocessed analytic protein table, ICD-3 first-occurrence table, follow-up dates and filtered chromosome-level carriers under the configured data root. Main mediation additionally requires a **pre-existing runset and its individual-level trio tables**. Their generators are deliberately not added. The optional `04_mediation/00_aggregate_feasibility_results.sh` only summarizes existing feasibility chunks and does not generate a runset.

See `docs/input_contracts.md` for columns, sample identifiers and relative manifest paths. UK Biobank individual-level inputs must be obtained through approved access and prepared locally. This release includes no individual-level study data or downloaded annotation databases.

## Sequential execution

Run from any working directory; each launcher locates its Python program relative to itself. On a single local process, the mediator–outcome launchers cover all protein columns and the outcome/all-protein programs use one chunk.

```bash
bash 01_E_to_M/01_cis_gene_based_pqtl.sh
bash 01_E_to_M/02_merge_and_complete_FDR_correct.sh
bash 01_E_to_M/03_filter_rank_cis_pqtl_significant.sh

bash 02_M_to_Y/01_prevalent_proteom_phenome.sh
bash 02_M_to_Y/02_incident_proteom_phenome.sh
bash 02_M_to_Y/03_merge_and_fdr_bonf.sh
bash 02_M_to_Y/04_clean_prev_fdr_results.sh
bash 02_M_to_Y/05_clean_inc_fdr_results.sh

bash 03_E_to_Y/01_prepare_EY_matrix_file.sh
bash 03_E_to_Y/02_prepare_gene_disease_analysis_input.sh
bash 03_E_to_Y/03_E_to_Y_main_analysis.sh
bash 03_E_to_Y/04_merge_all_and_correction.sh

# The next stage requires the externally prepared mediation runset/tables.
bash 04_mediation/01_run_mediation_primary.sh
bash 04_mediation/02_aggregate_mediation_results.sh
bash 04_mediation/03_prioritize_mediation_hits.sh
bash 04_mediation/04_extract_main_and_supp_tables.sh

bash 05_all_protein_comparison/01_build_all_protein_mediation_runset.sh
bash 05_all_protein_comparison/02_run_all_protein_mediation_scan.sh
bash 05_all_protein_comparison/03_aggregate_rank_all_protein_mediators.sh
bash 05_all_protein_comparison/04_compare_encoded_vs_best_nonencoded.sh
bash 05_all_protein_comparison/05_encoded_nonencoded_correlation.sh
bash 05_all_protein_comparison/06_conditional_mediation_encoded_vs_nonencoded.sh
bash 05_all_protein_comparison/07_residual_mediation_analysis.sh
bash 05_all_protein_comparison/08_final_pathway_classification.sh

bash 06_enrichment/01_prepare_protein_sets.sh
bash 06_enrichment/02_run_enrichment_ora.sh
```

`MODE=prevalent` or `MODE=incident` restricts the shared outcome/merge launchers to one outcome type; their default is both. The E→Y matrix now precedes its count/QC preparation. All-protein runset construction accepts the supported mediation table directly, so it does not require biological annotation or case-card scripts.

The default enrichment input is the Tier1/Tier2 union, matching the final universe launcher. `ENRICHMENT_SETS="tier1 tier2 combined"` also permits explicitly requested subset runs. The preparation stage persists the sorted unique union as a file; no `eval` or process-substitution input is used.

## Slurm and chunk completeness

Chromosome arrays are **1-based**, mapping to the configured `CHROMOSOMES` list. Other analysis arrays are **0-based**. An example preserving the original 50-task all-protein allocation is:

```bash
sbatch --array=0-49 --export=ALL 05_all_protein_comparison/02_run_all_protein_mediation_scan.sh
# Wait until all 50 tasks have completed successfully, then merge:
N_CHUNKS=50 bash 05_all_protein_comparison/03_aggregate_rank_all_protein_mediators.sh
```

Alternatively set `CHUNK_ID` and `N_CHUNKS` explicitly for a non-Slurm executor. Mediator–outcome arrays retain 60 proteins per task; the launcher checks that the declared number of tasks covers the protein header. A single task covers the whole panel. Mergers reject missing or unexpected numbered chunk files. Use a new results directory for a new input/configuration batch; existing same-numbered chunks cannot be assumed to come from the same input without checking their run records.

Direct Python interfaces remain available using `--help`. Launchers append any supplied CLI arguments to their configured defaults. Use the configuration for paths used in launcher preflight checks and chunk validation.

## Models and random numbers

The original model fitting functions and default scientific thresholds are retained. Model details and exceptions are in `docs/model_settings.md`. Primary mediation uses 200 bootstrap draws, at least 100 successful draws, a stability threshold of 0.80 and base seed 20260416. The per-trio seed remains the original MD5-derived value, independent of task allocation. All-protein/conditional/residual scans retain their original analytic Delta/Sobel approximations.

Actual commands, configuration hashes, input file metadata, small-file hashes, Python/package versions and exit status are written under `LOG_ROOT`. Numerical defaults are explicit in `config.example.sh` and each Python `--help`; some reporting constants remain fixed at the study value 0.05 as documented.

## Enrichment: universe and GO namespaces

The universe is the union of **all members of the supplied GMT**, before term-size filtering. No measured-protein background is supplied. The existing one-sided Fisher test, term-size limits 5–500, exclusion of zero-overlap terms and BH correction over returned tested terms are preserved.

For the complete GO GMT, BP/MF/CC are labelled **after one joint FDR calculation**. Outputs include `combined.go.ora.tsv`, `combined.go.ora.BP.tsv`, `combined.go.ora.MF.tsv` and `combined.go.ora.CC.tsv`. The three split tables retain the joint `fdr_q` values; no within-namespace recalculation occurs. Original columns are preserved, with `ontology`, `background_method` and `fdr_scope` appended. Unknown prefixes remain in the joint result as `Unclassified` and are counted in QC.

Use the same historical GMT files as the study. Sources, versions and acquisition instructions are in `docs/external_resources.tsv`; the runtime log hashes small input files including GMT resources.

## Validation and exclusions

`docs/workflow.tsv` maps retained programs to inputs, outputs and upstream steps. `docs/validation.md` describes synthetic execution and numerical regression checks. No UKB analysis has been rerun for this release, and exact Python 3.12.6/R 4.5.1 runtime validation remains separate from the local tests.

This directory prepares code for review and archiving. A code licence, repository ownership and a DOI must be supplied through the later repository-publication process; they have not been inferred here.
