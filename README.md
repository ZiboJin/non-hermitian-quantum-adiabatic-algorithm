# Non-Hermitian Quantum Adiabatic Algorithm

Code and data for **Non-Hermitian Quantum Adiabatic Algorithm** by **Zi-Bo Jin and Yi Zhang**, [arXiv:2607.15343](https://arxiv.org/abs/2607.15343).

## Contents

| Figure | Data | Contents |
| --- | --- | --- |
| 5: success probability versus problem size | [data/figure5](data/figure5/README.md) | CK results; 2,000 CK-like graph instances; 6,000 algorithm results and their statistics |
| 6: pseudospectra and stability thresholds | [data/figure6](data/figure6/README.md) | Contours, 13 numerical grids, 5 numerical thresholds and 11 size-dependent theoretical thresholds |
| 7: success probability under noise | [data/figure7](data/figure7/README.md) | 4,500 individual realizations, 45 statistical summaries and baseline probabilities |

`code/` contains the computation and plotting modules. `scripts/` provides commands to reproduce the results. `assets/` contains the mathematical text labels used in Figure 6.

## Installation

Tested with Python 3.13.5. From the repository root:

```sh
conda env create -f environment.yml
conda activate nhqaa
```

Alternatively, create and activate a Python 3.13 virtual environment, then run:

```sh
python -m pip install -r requirements.txt
```

## Reproduce the figures

```sh
python scripts/reproduce_figures.py
```

This reads the supplied data and writes `figure5.pdf`, `figure6.pdf`, `figure7.pdf` and PNG previews to `reproduced/`. To select figures or an output directory:

```sh
python scripts/reproduce_figures.py --figures 5 7 --output-dir reproduced/selected
```

To check file integrity, graph instances, statistics, random seeds and pseudospectral grids:

```sh
python scripts/verify_data.py
```

## Run the calculations

```sh
python scripts/run_calculations.py figure5-ck
python scripts/run_calculations.py figure5-ck-like
python scripts/run_calculations.py figure6-grids
python scripts/run_calculations.py figure6-theory
python scripts/run_calculations.py figure7
```

Results are written to `recomputed/`. Use `--dry-run` to inspect the parameters or `--max-workers 4` to use four workers (default: one). Full simulations take substantially longer than redrawing the figures.

The CK-like calculation reads the exact graph instances in `data/figure5/graph_pool.jsonl`. Graph generation and reconstruction are implemented in `code/generate_ck_like_random_deletion_graphs.py`. Each data directory contains field definitions and calculation settings.

## Citation

```bibtex
@misc{jin2026nonhermitian,
  title         = {Non-Hermitian Quantum Adiabatic Algorithm},
  author        = {Zi-Bo Jin and Yi Zhang},
  year          = {2026},
  eprint        = {2607.15343},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2607.15343},
  url           = {https://arxiv.org/abs/2607.15343}
}
```

## License

Code and software documentation: [MIT](LICENSE). Data and figure-label assets: [CC BY 4.0](LICENSE-DATA.md).
