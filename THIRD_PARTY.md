# Software and scientific data provenance

This repository contains the project's sequence-model implementation and utilities for its released scientific results. The data assets retain source identities, training overlap records, model identities and checksums.

The sequence implementation uses PyTorch, PyTorch Lightning, NumPy, pandas, scikit-learn, Biopython, PyYAML and ESM. Upstream ESM-C, ESM2, PIPPack and comparator models retain the terms of their respective providers. Structural comparison records identify AlphaFold3, MHC-Fine, AF2-finetune, tFold-pMHC and PANDORA configurations.

The evaluation sources include IEDB, PDB, the MuNIS and MHC-Bench data, and published patient cohorts. The dataset-specific README files and provenance tables identify their scientific sources and processing. Patient records in this release use the identifiers supplied in the published source cohorts.

The release supplies prepared scientific inputs and evaluation artifacts used in the paper. Its original source metadata remains part of the archive. A repository-wide reuse licence has not been assigned. Users should consult the originating datasets and software for their respective terms.

