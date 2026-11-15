# Reproduction guide

## Install data

```bash
python scripts/fetch_release_data.py
python scripts/verify_data.py
```

The four versioned data assets are `PrismMHC_Data_1_2.zip`, `PrismMHC_Data_3_4.zip`, `PrismMHC_Data_5.zip` and `PrismMHC_Data_6.zip`. `SHA256SUMS.txt` verifies the downloaded archives. `RELEASE_MANIFEST.json` verifies installed files relative to `data/`. An interrupted download is retried on the next invocation.

`--data-dir` changes the installation directory. For example, to work with an already extracted collection, pass the directory containing `Data_1` through `Data_6` to the result script.

```bash
python scripts/reproduce_results.py --data-dir /path/to/supplementary_data --output outputs/results.json
```

## Numerical checks

`reproduce_results.py` supports `--section all`, `el`, `ba`, `structure` and `imm`. It prints a JSON report and optionally saves it to the specified output file. Checks fail on population or prediction mismatches.

- EL recomputes pooled AUROC, average precision and PPV@n from the fixed source-clean HLA63 rows and from the common 55-HLA population. Paired inference uses peptide clusters in the original stored protocol.
- BA recomputes Pearson, Spearman, MAE and binder classification metrics on the common 1,541-pair population. Binding labels use the measured IC50 equivalent of 500 nM on the 1-log50k scale.
- Structure recomputes median peptide RMSDs and paired Wilcoxon statistics on the same 37 complexes. Timing summarizes the stored matched warm prepared-input measurements.
- IMM replays the six-member probability combination for Mix, P and patient candidates from stored logits, checks every resulting score against the released scores and recomputes patient AP, Hits@k and Success@k. AP and Success@k use the 53 positive-bearing patients. Hits@k includes all 70 patients.

The source-table alias `PrismMHC` denotes PrismMHC-EL, and `BA-S2+D4` denotes the published PrismMHC-BA affinity ensemble.

The result script preserves each fixed evaluation population and uses the original frozen scores. Some detailed analysis scripts in the data archive retain source-machine paths for provenance. Their source inventories and manifests document the required inputs.

## Detailed uncertainty analyses

The existing paired bootstrap draws, protocols and implementations are provided with each dataset.

| Analysis | Location under `data/supplementary_data/` |
|---|---|
| EL paired peptide-cluster bootstrap | `Data_2/EL/pooled_tests_20261109/` |
| BA pooled correlation intervals | `Data_2/current_figure/ba/` |
| Structure, source exclusions and paired comparisons | `Data_3/deduplicated165/` and `Data_3/current_figure/analysis/` |
| Contact metrics and paired query-cluster intervals | `Data_4/contact/` and `Data_4/reproduction/verify_current.py` |
| IMM and patient paired statistics | `Data_6/expert_revision_20261116/` |

The portable Contact replay compiles `cluster_metrics.cpp` with a C++17 compiler if its local shared library is absent. It checks 1,000-resample interval calculations against the frozen tables.

```bash
python data/supplementary_data/Data_4/reproduction/verify_current.py
python data/supplementary_data/Data_6/reproduction/reproduce.py
```

The additional IMM uncertainty replay uses NumPy, pandas, SciPy and scikit-learn, all listed in `requirements-reproduction.txt`.

```bash
python data/supplementary_data/Data_6/expert_revision_20261116/replay_intervals.py
```

Its default mode reconstructs all 168 patient point differences, confidence intervals and permutation P values, including regeneration of the stored patient sampling indices. It recomputes point estimates for the 10 current panel endpoints and 16 historical P/Mix component contrasts, and verifies their intervals using the complete stored bootstrap metric arrays. It also regenerates initial pooled bootstrap samples to connect those arrays to the supplied predictions and seeds, covering 96 current-panel resamples and 16 historical-panel resamples in total. `INTERVAL_REPLAY_QA.json` reports those counts explicitly.

Full regeneration of the pooled bootstrap arrays is available through the following options. Each current panel has 5,000 draws and each historical P/Mix panel has 10,000 draws.

```bash
python data/supplementary_data/Data_6/expert_revision_20261116/replay_intervals.py --regenerate-current
python data/supplementary_data/Data_6/expert_revision_20261116/replay_intervals.py --regenerate-historical
```

The same directory contains `reproduce_tables.py` for the SI tables and Success@k bootstrap/McNemar calculations, `replay_paired_tests.py` for 84 exact AP/count tests, and `audit_tie_order.py` for the published versus peptide-lexical tie-order comparison. Its README describes the fixed inputs and sampling protocols.

## Figures and molecular source data

| Figure content | Source directories |
|---|---|
| Framework and training coverage | `Data_1/training_coverage_20261106/figure1_export/` |
| EL and BA comparisons | `Data_2/current_figure/`, `Data_2/EL/pooled_tests_20261109/` |
| Structure and inference timing | `Data_3/current_figure/`, `Data_3/structural_quality/` |
| Contact labels and decoding | `Data_4/figure/`, `Data_4/contact/` |
| pMHC atlas, TCR groups and attention | `Data_5/atlas_corrected_20261106/`, `Data_5/terminology_fig3c_20261109/` |
| Immunogenicity and patient ranking | `Data_6/figures/figure5_roc/` |

These directories contain scientific data, rendered assets and their generating scripts. Dated upstream figure material remains identified in the corresponding README files. Rendering molecular scenes may require the original structural visualization environment and fonts.

## Sequence-model implementation

The curated `pmhcuni` package and top-level training/prediction programs expose the sequence model and its data encoding. Install `requirements-models.txt` to use them. Supply the appropriate model checkpoint, HLA sequence table and dataset configuration. The current ensemble and training identities are defined by Data 1. Large upstream checkpoints and complete training corpora are identified through the original source registries.

