# PrismMHC

Reproduction code and scientific data for **PrismMHC: Connecting antigen presentation, molecular modelling and neoantigen prioritization through protein language model adaptation**.

PrismMHC adapts protein language models with peptide–HLA supervision and uses the resulting representations for presentation prediction, binding-affinity estimation, peptide modelling, contact decoding and immunogenicity assessment.

The release includes fixed benchmark predictions, evaluation scripts, scientific source data, Contact decoder weights and the six-member IMM model with fitted preprocessing. Large upstream model checkpoints are identified by the supplied registries. Availability and input requirements for each module are described in [MODEL_AVAILABILITY.md](MODEL_AVAILABILITY.md).

## Reproduce the reported results

The CPU result checks require Python 3.11 and the packages in `requirements-reproduction.txt`.

```bash
python -m pip install -r requirements-reproduction.txt
python scripts/fetch_release_data.py
python scripts/verify_data.py
python scripts/reproduce_results.py --output outputs/results.json
```

The downloader uses the versioned GitHub release `manuscript-2026-11-16`, verifies the asset checksums and installs the data under `data/supplementary_data/`. The result checks recompute EL/BA metrics, the paired structural summary and patient ranking, and replay IMM probability aggregation from stored member logits. Detailed bootstrap and figure scripts are included with the corresponding scientific data.

See [REPRODUCTION.md](REPRODUCTION.md) for individual commands, exact data populations and the figure-source index.

## Code layout

- `scripts/` contains the portable release downloader, file verifier and numerical reproduction entry point.
- `pmhcuni/` contains the sequence-model, data, pooling and interpretation implementations used in the project.
- `training.py`, `training_v2.py`, `predict.py`, `evaluate.py` and `explain.py` expose the sequence-model entry points. These require the extended environment and the appropriate checkpoint and input files.
- `data/supplementary_data/Data_1` through `Data_6`, installed from the release, contain model registries, benchmark scores, source tables, analysis code and module-specific model artifacts.

The release preserves model-specific training and evaluation identities. EL uses its five-member probability ensemble, BA uses its specified affinity ensemble, and IMM averages six member probabilities before taking the patient candidate's maximum over supplied HLA assignments.

## Attribution and terms

Scientific source datasets and third-party dependencies are identified in the release registries and the manuscript references. See [THIRD_PARTY.md](THIRD_PARTY.md) for the software and dataset provenance. A repository-wide reuse licence has not been assigned in this release.

