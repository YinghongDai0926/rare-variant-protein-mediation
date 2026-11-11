# Validation record
The retained analysis code was tested with **synthetic data only**. No UK Biobank individual-level input was copied or analysed.
## Runtime
- Author-reported analysis runtime: Python **3.12.6**; R **4.5.1** (2025-06-13), Great Square Root.
- Actual local execution runtime: Python **3.11.17**, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, statsmodels 0.15.0. The available local R is 4.6.1 and was not invoked by these core Python programs.
- Exact-runtime validation in Python 3.12.6 / R 4.5.1 has **not** been performed; the supplied environment specification records the intended versions rather than claiming an installed, solved or frozen environment.
## Synthetic integration
800 artificial participants, two carrier genes, four protein columns and artificial baseline/incident dates were used. To keep this fixture small, the test-only configuration used 40 bootstrap draws, at least 20 successful draws, a 200-person incident sample gate and chromosome 1. The release defaults remain 200 draws, 100 successes, a 2000-person incident sample gate and chromosomes 1–22/X/Y.
The 26 sequential analysis/enrichment launchers passed. The optional existing feasibility aggregator also passed on a separate synthetic manifest. A chromosome-array task with 1-based ID=1/count=1 passed, checking the chromosome/zero-based chunk distinction.
| Entrypoint | Return code | Expected |
|---|---:|---:|
| `01_E_to_M/01_cis_gene_based_pqtl.sh` | 0 | 0 |
| `01_E_to_M/02_merge_and_complete_FDR_correct.sh` | 0 | 0 |
| `01_E_to_M/03_filter_rank_cis_pqtl_significant.sh` | 0 | 0 |
| `02_M_to_Y/01_prevalent_proteom_phenome.sh` | 0 | 0 |
| `02_M_to_Y/02_incident_proteom_phenome.sh` | 0 | 0 |
| `02_M_to_Y/03_merge_and_fdr_bonf.sh` | 0 | 0 |
| `02_M_to_Y/04_clean_prev_fdr_results.sh` | 0 | 0 |
| `02_M_to_Y/05_clean_inc_fdr_results.sh` | 0 | 0 |
| `03_E_to_Y/01_prepare_EY_matrix_file.sh` | 0 | 0 |
| `03_E_to_Y/02_prepare_gene_disease_analysis_input.sh` | 0 | 0 |
| `03_E_to_Y/03_E_to_Y_main_analysis.sh` | 0 | 0 |
| `03_E_to_Y/04_merge_all_and_correction.sh` | 0 | 0 |
| `04_mediation/01_run_mediation_primary.sh` | 0 | 0 |
| `04_mediation/02_aggregate_mediation_results.sh` | 0 | 0 |
| `04_mediation/03_prioritize_mediation_hits.sh` | 0 | 0 |
| `04_mediation/04_extract_main_and_supp_tables.sh` | 0 | 0 |
| `05_all_protein_comparison/01_build_all_protein_mediation_runset.sh` | 0 | 0 |
| `05_all_protein_comparison/02_run_all_protein_mediation_scan.sh` | 0 | 0 |
| `05_all_protein_comparison/03_aggregate_rank_all_protein_mediators.sh` | 0 | 0 |
| `05_all_protein_comparison/04_compare_encoded_vs_best_nonencoded.sh` | 0 | 0 |
| `05_all_protein_comparison/05_encoded_nonencoded_correlation.sh` | 0 | 0 |
| `05_all_protein_comparison/06_conditional_mediation_encoded_vs_nonencoded.sh` | 0 | 0 |
| `05_all_protein_comparison/07_residual_mediation_analysis.sh` | 0 | 0 |
| `05_all_protein_comparison/08_final_pathway_classification.sh` | 0 | 0 |
| `06_enrichment/01_prepare_protein_sets.sh` | 0 | 0 |
| `06_enrichment/02_run_enrichment_ora.sh` | 0 | 0 |
| `02_M_to_Y/03_merge_and_fdr_bonf.sh` | 1 | 1 |

The final mediator–outcome merge entry is an intentional missing-chunk rejection test. All normal analysis entries returned 0.
## Numerical preservation and I/O checks
- The universe ORA function was compared against the original final universe implementation on the same synthetic sets: the original result columns, including every P and q value, were **exactly equal**.
- BP/MF/CC split outputs were verified to retain the parent table's joint FDR q values; no within-namespace correction was performed.
- Legacy blank chunk input was tested together with a valid chunk; merging completed and retained the valid row.
- Missing/unexpected chunks are rejected before global correction; intentional rejection returned 1.
- 29 statistical helper function bodies (model fits, bootstrap, stable seed, residualization, indirect-effect calculations, BH and ORA) matched the original AST after excluding docstrings.
- 27 Python AST checks and 29 Bash syntax checks passed; existing core CLI help interfaces were checked, with the optional tqdm fallback corrected and rechecked.

## Issues corrected during synthetic execution
- Quoted carrier-template braces are preserved correctly in configuration.
- A missing optional tqdm dependency no longer prevents logistic analysis; it only suppresses the progress display.
- The original incident minimum sample threshold (2000) is exposed and documented without changing its default.
- Empty outcome chunks retain their schema; mergers handle legacy blank files and batches with no skipped records.
- Relative mediation table paths work through the configured table root.

These checks establish portable execution on the fixture and preservation of tested calculations. They do not validate study effect estimates, historical dependency versions, externally supplied candidate/runset generation, or HPC-scale performance.
