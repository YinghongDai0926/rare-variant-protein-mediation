# External input contracts

| Config variable / file | Required structure | Origin and consumer |
|---|---|---|
| `ANALYTIC_TSV` | TSV: `sample`, `age`, `sex`, `PC1`–`PC10`, then numeric protein columns named by gene/protein symbol | Externally preprocessed UKB analytic cohort; all three association stages, all-protein comparison |
| `ICD_FIRST_TSV` | TSV: `sample`, `icd3`, `first_date` | Externally derived first occurrence; date parseable as ISO YYYY-MM-DD; outcome association and all-protein analysis |
| `FOLLOWUP_TSV` | TSV: `sample`, `baseline_date`, `death_date`, `censor_date` | Externally derived baseline/death/censoring; missing death dates permitted by original logic |
| `CARRIERS_TSV_TEMPLATE` | One chromosome TSV: `gene`, `sample`, `carrier`; carriers represented by positive/1 values | Externally filtered rare deleterious carriers; `{CHR}` is replaced with 1–22/X/Y by each program |
| `MEDIATION_RUNSET_TSV` | TSV: `trio_id`, `gene`, `protein`, `icd3`, `tier_label`, `table_path` | Existing candidate/feasibility process, not generated in this subset |
| Mediation trio tables | TSV: `E`, `M`, `event`, `time_days`, `eligible_for_incident`, age/sex/PC covariates; `sample` recommended for provenance | Precomputed individual-level complete-case source for a particular trio |
| Optional feasibility chunks | Existing `01_feasibility.chunkN.tsv` and skipped counterparts with original feasibility metrics | Input to existing 00 aggregator; its producing scans and runset generators are excluded |
| `REACTOME_GMT`, `GO_GMT` | UTF-8 GMT: term identifier, description/URL, then gene symbols | MSigDB historical human-symbol collections; see resources manifest |

Sample IDs must match across inputs and must not be published with the code. Use consistent gene/protein symbols and ICD-3 codes. The association programs retain the original sample alignment and missing-value policies; these differ by stage as recorded in `model_settings.md`.

E→Y preparation generates `E_matrix.carrier_binary.npz` (samples by genes), `E_matrix.burden_counts.npz`, `E_matrix.samples.txt`, `E_matrix.genes.txt` and `E_matrix.carrier_counts.tsv`. Matrix row/column order is defined by the corresponding text files. Carry these together to all-protein comparison.

Relative `table_path` values are resolved under `MEDIATION_TABLE_ROOT`. Without `--table_root`, direct Python invocation uses the runset directory. Existing absolute manifest paths remain supported. Set both `MEDIATION_PATH_PREFIX_FROM` and `MEDIATION_PATH_PREFIX_TO` for an explicit prefix remap; the suffix is preserved, so a renamed root does not flatten subdirectories or silently guess a basename.

The release does not infer Tier1/Tier2 candidate definitions, regenerate phenotypes, reannotate variants or recreate missing runsets. Those inputs remain the responsibility of the existing study workflow.

Outputs are routed to configured stage directories. Empty mediator–outcome results retain column headers; the merger also tolerates legacy blank files. Other pre-existing safe readers remain in place. Failed, skipped and empty tasks should be distinguished using run logs and the programs' QC/skipped outputs.
