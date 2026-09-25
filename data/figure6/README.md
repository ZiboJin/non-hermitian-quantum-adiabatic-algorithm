# Figure 6: pseudospectra and stability thresholds

| Panels | Files |
| --- | --- |
| (a)-(d) | `contours.npz` and `contours_metadata.csv`: 13 contours indexed by `contour_key` |
| (e), numerical | `thresholds_numerical.csv`: 5 gap-closing thresholds for `r=1,...,5` |
| (e), theory | `thresholds_theory.csv`: estimates from the MIS sector |
| (f) | `thresholds_vs_size.csv`: 11 theoretical thresholds at `n=5,9,...,45` |
| Underlying FK grids | `grids/`: 8 local 161-by-161 grids and 5 pair grids |
| Grid settings | `local_grid_manifest.csv` and `pair_grid_manifest.csv` |

## Settings

Panels (a)-(e) use CK `m=2`, `n=5`, `p=2`, `q=4`, with contour level `epsilon=1e-32`. FK has `L=14*r`. The HD/HM local block uses `theta=pi/4`, `Omega=1`.

FK singular values use 70 decimal digits and at most 40 inverse-iteration steps. Pair grids have 81 points per axis for `r=1,...,4` and 241 for `r=5`. Local grids have 161 points per axis. Both window expansion factors are 2. HD/HM contours use 90 decimal digits and 241 angles.

Panel (f) uses `m=2,...,12`, `r=n=4*m-3` and the MIS sector. All displayed theoretical thresholds select the ground and first excited levels.

## Arrays and coordinates

Read NPZ files with `numpy.load(path, allow_pickle=False)`.

- Each contour array has two columns. `contour_coordinate_frame` specifies local displacement `(Re(z-E_j), Im(z-E_j))`, displacement divided by epsilon for HD/HM, or absolute `(Re z, Im z)` for the connected FK contour.
- Panels (b) and (c) use the radial transformation `asinh(rho/1e-29)` while preserving the polar angle.
- Grid arrays `xs` and `ys` contain the axes. `logF[i,j]` is `log10(min_x sigma_min(z_ij*I-H_x))`, with rows along `ys` and columns along `xs`.
- `local_coordinates=True` indicates coordinates relative to the stored center. Pair grids use absolute coordinates.
- `cache_path` is relative to the repository root. `source_cache_path` records the original calculation path.
- High-precision center and eigenvalue fields are decimal strings; retain their precision when loading them.

## Threshold columns

`log10_epsilon_c_grid_hp` is the numerical threshold where the ground/first-excited components connect. `log10_epsilon_est_mis` is the theoretical estimate used in panel (e).

Panel (f) uses `log10_epsilon_threshold_theory`. The linear `epsilon_threshold_theory` column underflows to zero for `n>=13`; use the logarithmic values. `k_star_est` identifies the excited level used in the estimate.
