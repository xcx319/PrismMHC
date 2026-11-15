# Figure sources and numbering

The `figures-2026-11-16` release supplies ten independently numbered main figures, editable SVGs, embedded-font PDFs, 600-dpi PNGs, rendering code and frozen plotting inputs. The revision updates typography, spacing and panel organization using the scientific data from the manuscript reproduction release. [Download the figure source bundle](https://github.com/xcx319/PrismMHC/releases/tag/figures-2026-11-16).

## Reproduce the figures

Download `PrismMHC_figures_20261116.zip` and `FIGURE_BUNDLE_SHA256.txt` from the release, or use GitHub CLI.

```bash
gh release download figures-2026-11-16 --repo xcx319/PrismMHC \
  --pattern PrismMHC_figures_20261116.zip --pattern FIGURE_BUNDLE_SHA256.txt
sha256sum -c FIGURE_BUNDLE_SHA256.txt
unzip PrismMHC_figures_20261116.zip
cd PrismMHC_figures_20261116
python -m pip install -r requirements.txt
python render_all.py
```

Rendering uses Python 3.11, NumPy, pandas, SciPy, Matplotlib, PyMuPDF and Pillow. Install Arial regular and bold before rendering. The included PDFs already embed the native font subsets. The render entry point uses Matplotlib's noninteractive Agg backend and writes the ten figure outputs to `figures/`.

Figure 1 uses normal-width Arial and aligned framework components. Ligand presentation and binding affinity have independent figure numbers. The atlas, AS4.3 example and attention/contact analysis also have separate figure numbers. Immunogenicity screening and patient prioritization appear in Figures 9 and 10, with horizontal AP distributions.

## Figure map

| Current figure | Topic | Earlier figure panels |
|---|---|---|
| 1 | Framework | 1a–c |
| 2 | Ligand presentation | 2a–g |
| 3 | Binding affinity | 2h–k |
| 4 | Representation atlas | 3a |
| 5 | AS4.3 structure and attention | 3b |
| 6 | Attention/contact associations | 3c |
| 7 | Structure prediction | 4a–b |
| 8 | Contact decoding | 4c–h |
| 9 | IMM architecture and immunogenicity screening | 5a–g |
| 10 | Patient prioritization | 5h, 5i, 5k, 5j |

`FIGURE_MAP.json` records the output filenames and panel identities. The source bundle contains numerical preservation and layout checks alongside each rendering component. Six vector plates are assembled from frozen scientific PDFs with updated panel identities. EL and IMM figures are rendered from the supplied curve and observation tables. The framework drawing uses schematic elements and the existing 7MKB molecular illustration.

`MANIFEST.json` records the shipped file sizes and SHA256 hashes before local rerendering. The numerical checks verify EL data and curves, 1,800 IMM AP observations, 36 distribution summaries and 84 IMM curve paths. The original benchmark data remain available in [the manuscript reproduction release](https://github.com/xcx319/PrismMHC/releases/tag/manuscript-2026-11-16).
