# Model artifacts and input requirements

| Module | Artifacts supplied in the data release | Requirements for new inputs |
|---|---|---|
| PrismMHC-EL | Frozen benchmark scores, ensemble specification, training-source and checkpoint registries, pooling/training implementation | The five selected sequence-model checkpoints and their backbone runtimes. The checkpoint registry identifies original files and hashes. |
| PrismMHC-BA | Frozen common-population predictions, affinity ensemble specification, paired evaluation source data | The selected affinity-model checkpoints and feature producers identified in Data 1. |
| PrismMHC-AA | Final-model coordinate arrays, per-complex errors, paired runtime measurements, source exclusions and implementation records | The complete final structural pipeline, adapted sequence features, scaffold inputs and registered external encoder/checkpoint dependencies. |
| PrismMHC-Contact | Thirty compact trained decoder checkpoints, normalization arrays, decoder implementation, prepared-feature replay fixture and frozen contact predictions | Attention and sequence descriptors from the corresponding upstream backbones. |
| PrismMHC-IMM | Six trained fusion members, fitted preprocessing, model implementation, frozen member logits, pair tables and patient HLA assignments | The six standardized feature views, including upstream EL/BA scores, sequence representations and structural graph features. |

The CPU reproduction commands operate on the fixed published predictions and model logits. They require no large backbone download or GPU. Prepared-feature Contact and IMM inference additionally requires PyTorch and the relevant feature tensors.

## Model identity

The current EL specification averages five member probabilities. BA combines affinity predictions on the continuous transformed-affinity scale using its registry weights. These identities are recorded in `Data_1/EL5_registry.json`, `Data_1/BA_registry.json` and their associated implementation records. The comprehensive sequence checkpoint inventory is `Data_1/training_coverage_20261106/checkpoint_inventory.json`.

The structure model is identified in `Data_3/deduplicated165/MODEL_MANIFEST.json`, with final checkpoint ID S250 step512 and SHA256 `1bf8bdcc7135d5f45086977660e86ac7ed424445dc84748a203a8fe146083e74`. Its current structural benchmark has 37 common complexes. Data 3 also retains source-deduplicated evaluations from the 165-complex collection with individual inclusion flags.

The Contact decoder specification is `Data_4/model/release.json`. It enumerates ten feature-producer groups and three decoder seeds per group, together with model and normalization checksums.

The IMM specification is `Data_6/ensemble.json`. Member probability weights are 0.20, 0.40, 0.10, 0.10, 0.10 and 0.10. The first five use a bounded tanh residual and the sixth uses a bounded softsign residual. The ensemble scores each peptide–HLA pair before patient-level HLA maximization. Prepared-feature inference is provided by `Data_6/reproduction/imm_ensemble.py`.

Checkpoint names such as v11 or v12 occur in several component lineages. The release registry and hashes identify the role of each checkpoint. The standalone sequence entry points accept explicit checkpoints and do not automatically reconstruct every manuscript ensemble.

