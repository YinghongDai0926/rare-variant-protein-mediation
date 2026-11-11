# Preserved study settings

| Stage | Model / selection | Original defaults preserved |
|---|---|---|
| E→M | Covariate-residualized OLS equivalent to protein ~ carrier + age + sex + PC1–PC10; per-protein available samples | ≥10 carriers, ≥50 samples; age median, sex mode, missing PCs=0; gene/protein names must match for cis pairing |
| E→M correction | Global BH and Bonferroni over merged chromosome tests | Threshold 0.05; original coefficient output precision retained |
| M→Y prevalent | Logistic regression with age/sex/PC1–PC10 | ≥50 cases and ≥200 controls; original convergence/complete-case behavior |
| M→Y incident | statsmodels PHReg, original ties/default behavior | ≥50 events and ≥2000 complete-case at-risk samples per protein; exclude baseline-prevalent cases and use original follow-up/censoring definitions |
| M→Y correction/cleaning | Separate global correction for prevalent and incident; original anomaly rules | beta abs cutoff 10; SE cutoff 5; log effect tolerance 0.005; CI tolerance 0.05; underflow z=30 |
| E→Y matrix | Combined carrier matrix from supplied filtered carriers | ≥10 carriers, ≥200 noncarriers, max carrier rate 0.05 |
| E→Y association | Original logistic/GLM fallback and Breslow Cox | Global carriers≥5, max rate=0.2; cases/events≥20, controls≥20; used carriers≥3, used noncarriers≥30; carrier cell counts≥1; max abs beta=20 |
| Primary mediation | OLS a-path; Efron Cox total/direct models; product a*b on log-HR scale | CC≥200, events≥20, carriers≥10, noncarriers≥200, carrier events≥2, carrier nonevents≥5; mediator SD≥1e-8 |
| Primary bootstrap | Nonparametric resampling and original sign-based empirical two-sided P; percentile CI | 200 draws, ≥100 successes; base seed 20260416; per-trio MD5-derived seed; stable rate≥0.80; BH-FDR<0.05 |
| All-protein scan | Same outcome pair across measured protein panel; analytic Delta/Sobel approximation; Efron Cox | Same original sample/event/carrier gates; encoded protein exact symbol matching |
| Correlation | Original Pearson/Spearman and covariate-residual correlations | CC≥200; absolute correlation classes 0.7 and 0.4 |
| Conditional/residual | Existing models and analytic indirect-effect tests | Nominal retained P<0.05; original FDR outputs retained separately; model success and classification order unchanged |
| Universe ORA | GMT union universe; Fisher greater; original returned-term BH family | Term size 5–500; zero-overlap terms excluded as in supplied code; GO BP/MF/CC jointly corrected before namespace split |

The same-named study concepts can have different gates in screening and mediation. These are retained rather than harmonized. Sensitivity analyses and their R models are excluded by scope.

Some fixed reporting flags in all-protein comparison use FDR=0.05; changing nominal retained P does not change those FDR flags. The nominal summaries now follow the supplied `retained_p_cutoff`, with the same 0.05 default. Enrichment QC counts explicitly labelled 0.05 remain counts at 0.05; q values are threshold-independent. Original phenotype cutoffs and upstream rare-variant annotation rules are not re-created here.
